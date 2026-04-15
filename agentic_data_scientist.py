# Orchestrator for an "agentic" offline data scientist pipeline.
# Handles dataset loading, profiling, planning, training, evaluation, reflection,
# and optional re-planning cycles. Designed primarily for classification tasks.
import os
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pandas as pd  # type: ignore

# Agent components and tooling used by the orchestrator
from agents.planner import create_plan
from agents.reflector import reflect, should_replan, apply_replan_strategy
from agents.memory import JSONMemory
from tools.data_profiler import profile_dataset, infer_target_column, dataset_fingerprint
from tools.modelling import build_preprocessor, select_models, train_models, feature_selection
from tools.evaluation import evaluate_best, write_markdown_report, save_json


# Lightweight container for run metadata and parameters
@dataclass
class RunContext:
    run_id: str
    started_at: str
    data_path: str
    target: str
    output_dir: str
    seed: int
    test_size: float
    max_replans: int


def now_iso() -> str:
    """Return current UTC time in ISO 8601 format (no microseconds) with Z suffix."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class AgenticDataScientist:
    """
    Offline Agentic Data Scientist (classification-focused).

    Responsibilities:
    - load and profile datasets
    - create a plan (via planner)
    - build preprocessors and select candidate models
    - train and evaluate models
    - reflect on results and optionally re-plan
    - persist artefacts and update memory
    """

    def __init__(self, memory_path: str = "agent_memory.json", verbose: bool = True):
        # Verbose controls logging output
        self.verbose = verbose
        # Simple persistent memory used to remember prior runs for a dataset fingerprint
        self.memory = JSONMemory(memory_path)

        # Context and transient state populated when run() is executed
        self.ctx: Optional[RunContext] = None
        self.state: Dict[str, Any] = {}
        self.state['internal_memory'] = {}

        self.step_registry = {
            # "P1B_profile_dataset": self._step_profile_dataset,
            "P2B_build_preprocessor": self._step_build_preprocessor,
            "P3A_regularization": self._step_apply_regularization,
            "P3A_imb_class_weight": self._step_consider_imbalance_strategy,
            "P3A_feature_selection": self._step_feature_selection,
            "P3A_simpler_models": self.step_select_simpler_models,
            "P3A_decrease_model_complexity": self.step_decrease_model_complexity,
            "P3A_increase_model_complexity": self.step_increase_model_complexity,
            "P3B_select_models": self._step_select_models,
            "P4A_tune_hyperparameters": self._step_tune_hyperparameters,
            "P4B_train_models": self._step_train_models,
            "P5B_evaluate": self._step_evaluate,
            "P6B_reflect": self._step_reflect,
            "P7B_write_report": self._step_write_report,
        }

    def log(self, msg: str) -> None:
        """Print a log message when verbose is enabled."""
        if self.verbose:
            print(f"[AgenticDataScientist] {msg}")

    def load_data(self, path: str) -> pd.DataFrame:
        """Load a CSV into a pandas DataFrame and log its shape."""
        self.log(f"Loading dataset: {path}")
        df = pd.read_csv(path)
        self.log(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")
        return df
    
    
    # def _step_profile_dataset(self, state):
    #     state["profile"] = profile_dataset(state["df"], self.ctx.target)
    #     return state


    
    def _step_build_preprocessor(self, state):
        state["preprocessor"] = build_preprocessor(state["profile"])
        return state
    
    def _step_select_models(self, state):
        state["candidates"] = select_models(state["internal_memory"], seed=self.ctx.seed)
        # self.log(f"Candidate models: {[n for n, _ in state['candidates']]}")
        return state
    
    def _step_train_models(self, state):
        state["results"] = train_models(
            df=state['df'],
            target=self.ctx.target,
            preprocessor=state["preprocessor"],
            candidates=state["candidates"],
            seed=self.ctx.seed,
            test_size=self.ctx.test_size,
            output_dir=self.ctx.output_dir,
            feature_selector=state['feature_selector'] if 'feature_selector' in state else None,
            verbose=self.verbose,
        )
        return state
    
    def _step_evaluate(self, state):
        state["eval_payload"] = evaluate_best(state["results"], output_dir=self.ctx.output_dir)
        return state

    def _step_reflect(self, state):
        state["reflection"] = reflect(
            dataset_profile=state["profile"],
            evaluation=state["eval_payload"]
        )
        return state
    
    def _step_write_report(self, state):

        save_json(os.path.join(self.ctx.output_dir, "eda_summary.json"), state['profile'])
        save_json(os.path.join(self.ctx.output_dir, "plan.json"), {"plan": state['plan']})
        save_json(os.path.join(self.ctx.output_dir, "metrics.json"), state['eval_payload'])
        save_json(os.path.join(self.ctx.output_dir, "reflection.json"), state['reflection'])
        save_json(os.path.join(self.ctx.output_dir, "history.json"), state['history'])

        write_markdown_report(
            out_path=os.path.join(self.ctx.output_dir, "report.md"),
            ctx=self.ctx,
            fingerprint=state["fp"],
            dataset_profile=state["profile"],
            plan=state["plan"],
            eval_payload=state["eval_payload"],
            reflection=state["reflection"],
        )
        return state
    
    def _step_apply_regularization(self, state):
        state['profile']['plan_notes']['regularization'] = "Applied regularization parameters to classifiers due to small dataset size."
        return state
    
    def _step_consider_imbalance_strategy(self, state):
        state['profile']['plan_notes']['imbalance_strategy'] = "Add class_weight = 'balanced' to classifiersdue due to detected imbalance (imbalance_ratio >= 3.0)."
        return state
    
    def _step_feature_selection(self, state):
        state['profile']['plan_notes']['feature_selection'] = "Applied SelectKBest feature selection strategy due to reflection suggestions."
        state['feature_selector'] = feature_selection()
        return state
    
    def step_select_simpler_models(self, state):
        SIMPLER_MODEL_MAP = {
            "RandomForest": "DecisionTree",
            "GradientBoosting": "DecisionTree",
            "SVC_RBF": "LinearSVM",
        }
        # Choose a simpler model based on the best performing model from evaluation results
        best_model = state['eval_payload']['best_metrics']['model']
        suggested = SIMPLER_MODEL_MAP.get(best_model)

        if suggested:
            state['profile']['plan_notes']['simpler_model_selection'] = f"Reflection suggested trying simpler model: {suggested} instead of {best_model}."
            state['profile']['plan_suggestions'].setdefault('add_models', []).append(suggested)
            self.log(f"Reflection suggests trying simpler model: {suggested} instead of {best_model}")
        else:
            state['profile']['plan_notes']['simpler_model_selection'] = f"Model {best_model} is already the most simple model or the simpler model is already exist in candidates but it performed worse. No simpler model will be added to the plan."
            self.log(f"No simpler model suggestion for best model: {best_model}")
        return state
    
    def step_decrease_model_complexity(self, state):
        state['internal_memory']['search_space'] = 'simple'
        self.log("Reflection suggests decreasing model complexity by using a simpler hyperparameter search space for candidate models.")
        self.log(f"[FIX]Updated internal memory with simpler search space: {state['internal_memory']['search_space']}")
        return state
    
    def step_increase_model_complexity(self, state):
        state['internal_memory']['search_space'] = 'complex'
        self.log("Reflection suggests increasing model complexity by using a more complex hyperparameter search space for candidate models.")
        return state
    
    def _step_tune_hyperparameters(self, state):
        if not state['internal_memory'].get('search_space'):
            state['internal_memory']['search_space'] = 'normal'
        self.log(f"Applying hyperparameter tuning with search space: {state['internal_memory']['search_space']}")
        # In a real implementation, this would modify the candidate models to include hyper
    

    def run(
        self,
        data_path: str,
        target: str,
        output_root: str = "outputs",
        seed: int = 42,
        test_size: float = 0.2,
        max_replans: int = 5,
    ) -> str:
        """
        Main orchestration entry point.

        Parameters:
        - data_path: path to the CSV dataset
        - target: target column name or 'auto' to infer
        - output_root: directory where outputs are stored (subdir will be created)
        - seed/test_size: training reproducibility and test split
        - max_replans: maximum number of times to re-plan and re-run

        Returns: path to the output directory for this run
        """
        # Create a unique run id and output directory for artefacts
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        output_dir = os.path.join(output_root, run_id)
        os.makedirs(output_dir, exist_ok=True)

        # Populate run context with parameters and metadata
        self.ctx = RunContext(
            run_id=run_id,
            started_at=now_iso(),
            data_path=data_path,
            target=target,
            output_dir=output_dir,
            seed=seed,
            test_size=test_size,
            max_replans=max_replans,
        )
        # Internal state used to track replanning attempts
        self.state['replan_count'] = 0

        # Load dataset into memory
        self.state['df'] = self.load_data(data_path)

        # If client requested auto target detection, infer it from data
        if target.strip().lower() == "auto":
            inferred = infer_target_column(self.state['df'])
            if not inferred:
                raise ValueError("Could not infer target column. Please provide --target <name>.")
            # Update context with inferred target name
            self.ctx.target = inferred
            self.state['target'] = inferred
            self.log(f"Inferred target: {inferred}")

        # Produce a dataset profile (EDA summary) and a fingerprint used for memory
        self.state['profile'] = profile_dataset(self.state['df'], self.state['target'])
        self.state['fp'] = dataset_fingerprint(self.state['df'], self.state['target'])

        # Look up previous runs for the same dataset fingerprint (memory hint)
        self.state['prev'] = self.memory.get_dataset_record(self.state['fp'])
        print(f"Dataset fingerprint: {self.state['fp']}")
        if self.state['prev']:
            self.log(f"Memory hit: previously best={self.state['prev'].get('best_model')} for fp={self.state['fp']}")

        # Create an initial plan informed by the profile and optional memory hint
        self.state['plan'] = create_plan(self.state['profile'], self.state['internal_memory'], memory_hint=self.state['prev'])
        self.log(f"Plan: {self.state['plan']}")

        self.state['history'] = {}

        while True:
            state = self.state 

            for step_name in state['plan']:
                if step_name not in self.step_registry:
                    # raise ValueError(f"Unknown step: {step_name}")
                    print(f"Warning: No implementation for step '{step_name}'. Skipping.")
                    continue

                step_fn = self.step_registry[step_name]

                try:
                    state = step_fn(state)
                except Exception as e:
                    print(f"Step failed: {step_name} → {e}")
                    raise 

            self.state = state 

            # Log iteration info, including plan, profile, evaluation results, reflection, and replan decision
            iter_info = {
                'plan': self.state['plan'], 
                'plan_notes': self.state['profile']['plan_notes'], 
                'observation': {
                    "best_model": self.state['eval_payload']["best_metrics"]["model"],
                    "best_metrics": self.state['eval_payload']["best_metrics"]
                }, 
                'reflection': self.state.get('reflection'),
                'should_replan': should_replan(self.state['reflection']),
                'internal_memory': self.state['internal_memory'],
            }
            self.state['history'][f'iter_{self.state["replan_count"]}'] = iter_info

            # Update the memory store with outcomes from this run
            self.memory.upsert_dataset_record(self.state['fp'], {
                "last_seen": now_iso(),
                "target": self.ctx.target,
                "shape": self.state['profile']["shape"],
                "best_model": self.state['eval_payload']["best_metrics"]["model"],
                "best_metrics": self.state['eval_payload']["best_metrics"],
            })

            # Decide whether the agent should attempt to re-plan and re-run
            if not should_replan(self.state['reflection']):
                # No replan suggested — finish the run
                break

            # If we've already replanned the allowed number of times, stop
            if self.state["replan_count"] >= self.ctx.max_replans:
                self.log(f"Replan suggested, but max_replans reached:{self.ctx.max_replans}. Stopping.")
                break
           
            # Otherwise, increment replan counter and apply the replan strategy
            self.state["replan_count"] += 1
            self.log(f"Replanning attempt #{self.state['replan_count']}...")

            # apply_replan_strategy returns an updated (plan, profile) pair
            self.state['plan'], self.state['profile'] = apply_replan_strategy(
                self.state['plan'],
                self.state['profile'],
                self.state['reflection']
            )
            self.log(f"Re-Plan: {self.state['plan']}")

        # Final log and return the directory containing run outputs
        self.log(f"Done. Outputs saved to: {self.ctx.output_dir}")
        return self.ctx.output_dir
    


