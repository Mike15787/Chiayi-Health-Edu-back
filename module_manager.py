# module_manager.py
import importlib
import logging
import os
from typing import Dict, Type, Any

logger = logging.getLogger(__name__)

class ModuleManager:
    """
    Dynamically loads and manages education modules.
    Each module provides its own patient agent logic, scoring logic, and summary logic.
    """
    def __init__(self):
        self.modules: Dict[str, Dict[str, Any]] = {}
        self._load_all_modules()

    def _load_all_modules(self):
        """
        Scans the 'modules' directory and loads all detected modules.
        Each module directory should contain at least:
        - config.py (with MODULE_ID)
        - patient_agent.py (with build_patient_prompt function)
        - scoring_logic.py (with a ScoringLogic class matching module name)
        - summary_logic.py (with generate_summary_prompt and calculate_organization_efficiency_score_llm functions)
        """
        modules_root = os.path.join(os.path.dirname(__file__), 'scenarios')
        if not os.path.exists(modules_root):
            logger.error(f"Modules root directory not found: {modules_root}. Please create it.")
            return

        for module_dir_name in os.listdir(modules_root):
            module_path = os.path.join(modules_root, module_dir_name)
            if os.path.isdir(module_path) and not module_dir_name.startswith('__'):
                try:
                    # Dynamically import module components
                    config_module = importlib.import_module(f'scenarios.{module_dir_name}.config')
                    patient_agent_module = importlib.import_module(f'scenarios.{module_dir_name}.patient_agent')
                    scoring_logic_module = importlib.import_module(f'scenarios.{module_dir_name}.scoring_logic')
                    summary_logic_module = importlib.import_module(f'scenarios.{module_dir_name}.summary_logic')
                    precomputation_logic_module = importlib.import_module(f'scenarios.{module_dir_name}.precomputation_logic')

                    module_id = getattr(config_module, 'MODULE_ID')
                    
                    # Construct the class name dynamically (e.g., ColonoscopyBowelPrepAScoringLogic)
                    scoring_class_name = ''.join(word.capitalize() for word in module_dir_name.split('_')) + 'ScoringLogic'

                    self.modules[module_id] = {
                        'config': config_module,
                        'patient_agent_builder': getattr(patient_agent_module, 'build_patient_prompt'),
                        'scoring_logic_class': getattr(scoring_logic_module, scoring_class_name),
                        'summary_generator': getattr(summary_logic_module, 'generate_summary_prompt'),
                        'org_efficiency_scorer': getattr(summary_logic_module, 'calculate_organization_efficiency_score_llm'),
                        'precomputation_performer': getattr(precomputation_logic_module, 'perform_precomputation'),
                    }
                    logger.info(f"Loaded education module: {module_id}")
                except Exception as e:
                    logger.error(f"Failed to load module '{module_dir_name}': {e}", exc_info=True)
        
        if not self.modules:
            logger.warning("No education modules were loaded. Check 'modules' directory structure and module content.")

    def get_module(self, module_id: str) -> Dict[str, Any]:
        """Returns the loaded module components for a given module ID."""
        if module_id not in self.modules:
            # Fallback to a default module if necessary, or raise an error
            logger.error(f"Module '{module_id}' not found. Available modules: {list(self.modules.keys())}")
            raise ValueError(f"Module '{module_id}' not found. Please ensure it is correctly defined and loaded.")
        return self.modules[module_id]

    def get_patient_agent_builder(self, module_id: str):
        return self.get_module(module_id)['patient_agent_builder']

    def get_scoring_logic_instance(self, module_id: str):
        # Instantiate the scoring logic class for the specific module
        return self.get_module(module_id)['scoring_logic_class']()

    def get_summary_generator(self, module_id: str):
        return self.get_module(module_id)['summary_generator']
    
    def get_org_efficiency_scorer(self, module_id: str):
        return self.get_module(module_id)['org_efficiency_scorer']

    def get_module_config(self, module_id: str):
        return self.get_module(module_id)['config']
    
    def get_precomputation_performer(self, module_id: str):
        """返回指定模組的預計算函數。"""
        return self.get_module(module_id)['precomputation_performer']