Custom Integration Examples
===========================

This guide demonstrates how to integrate ScrambleBench with existing systems, custom evaluation pipelines, MLOps workflows, and enterprise infrastructure.

.. contents:: Table of Contents
   :local:
   :depth: 2

Embedding ScrambleBench in Existing Systems
--------------------------------------------

CI/CD Pipeline Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Integrate contamination testing into continuous integration workflows:

.. code-block:: yaml

   # .github/workflows/model-evaluation.yml
   name: Model Contamination Testing
   
   on:
     push:
       branches: [main]
     pull_request:
       branches: [main]
     schedule:
       - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM
   
   jobs:
     contamination-test:
       runs-on: ubuntu-latest
       
       strategy:
         matrix:
           model: 
             - "openai/gpt-3.5-turbo"
             - "anthropic/claude-3-sonnet"
             - "meta-llama/llama-2-70b-chat"
           complexity: [3, 5, 7]
       
       steps:
         - uses: actions/checkout@v3
         
         - name: Set up Python
           uses: actions/setup-python@v4
           with:
             python-version: '3.9'
         
         - name: Install ScrambleBench
           run: |
             pip install -e .
             pip install -r requirements-ci.txt
         
         - name: Run Contamination Analysis
           env:
             OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
           run: |
             python scripts/ci_contamination_test.py \
               --model "${{ matrix.model }}" \
               --complexity ${{ matrix.complexity }} \
               --benchmark-dir data/benchmarks/ci \
               --output-file results/contamination_${{ matrix.model }}_${{ matrix.complexity }}.json \
               --threshold 0.15
         
         - name: Upload Results
           uses: actions/upload-artifact@v3
           with:
             name: contamination-results
             path: results/

.. code-block:: python

   # scripts/ci_contamination_test.py
   """Contamination testing script for CI/CD pipelines."""
   
   import argparse
   import json
   import sys
   from pathlib import Path
   from typing import Dict, List, Any
   
   from scramblebench import TranslationBenchmark
   from scramblebench.llm import OpenRouterClient
   from scramblebench.translation.language_generator import LanguageGenerator, LanguageType
   from scramblebench.evaluation.metrics import ContaminationAnalyzer
   
   class CIContaminationTester:
       """Contamination tester optimized for CI/CD environments."""
       
       def __init__(self, threshold: float = 0.15):
           self.threshold = threshold  # Maximum acceptable contamination
           self.analyzer = ContaminationAnalyzer()
           
       def run_ci_test(
           self,
           model_name: str,
           complexity: int,
           benchmark_dir: Path,
           max_samples: int = 50  # Smaller for CI speed
       ) -> Dict[str, Any]:
           """Run contamination test optimized for CI environments."""
           
           # Initialize model
           model = OpenRouterClient(
               model_name=model_name,
               api_key=os.getenv('OPENROUTER_API_KEY'),
               timeout=30  # Shorter timeout for CI
           )
           
           # Generate test language
           language_generator = LanguageGenerator(seed=42)
           test_language = language_generator.generate_language(
               name=f"ci_test_complexity_{complexity}",
               language_type=LanguageType.PHONETIC,
               complexity=complexity,
               vocab_size=500  # Smaller vocabulary for speed
           )
           
           results = {}
           
           # Test each benchmark file
           for benchmark_file in benchmark_dir.glob("*.json"):
               benchmark_name = benchmark_file.stem
               
               try:
                   # Baseline evaluation
                   baseline_benchmark = TranslationBenchmark(
                       source_dataset=str(benchmark_file),
                       use_transformation=False
                   )
                   baseline_result = baseline_benchmark.run(model, num_samples=max_samples)
                   
                   # Transformed evaluation
                   transform_benchmark = TranslationBenchmark(
                       source_dataset=str(benchmark_file),
                       constructed_language=test_language,
                       preserve_structure=True
                   )
                   transform_result = transform_benchmark.run(model, num_samples=max_samples)
                   
                   # Calculate contamination metrics
                   contamination_score = baseline_result.score - transform_result.score
                   
                   results[benchmark_name] = {
                       'baseline_score': baseline_result.score,
                       'transformed_score': transform_result.score,
                       'contamination_score': contamination_score,
                       'passed': contamination_score <= self.threshold,
                       'complexity': complexity
                   }
                   
               except Exception as e:
                   results[benchmark_name] = {
                       'error': str(e),
                       'passed': False,
                       'complexity': complexity
                   }
           
           # Overall assessment
           all_passed = all(result.get('passed', False) for result in results.values())
           max_contamination = max(
               result.get('contamination_score', 0) 
               for result in results.values() 
               if 'contamination_score' in result
           )
           
           summary = {
               'model': model_name,
               'complexity': complexity,
               'threshold': self.threshold,
               'all_tests_passed': all_passed,
               'max_contamination_detected': max_contamination,
               'benchmark_results': results,
               'recommendation': self._get_recommendation(all_passed, max_contamination)
           }
           
           return summary
       
       def _get_recommendation(self, all_passed: bool, max_contamination: float) -> str:
           """Generate recommendation based on test results."""
           if all_passed:
               return "PASS: Model shows acceptable contamination levels for production use."
           elif max_contamination > self.threshold * 2:
               return "FAIL: Severe contamination detected. Do not deploy this model."
           else:
               return "WARNING: Moderate contamination detected. Consider additional validation."
   
   def main():
       parser = argparse.ArgumentParser(description='CI/CD Contamination Testing')
       parser.add_argument('--model', required=True, help='Model name to test')
       parser.add_argument('--complexity', type=int, default=5, help='Transformation complexity')
       parser.add_argument('--benchmark-dir', type=Path, required=True, help='Benchmark directory')
       parser.add_argument('--output-file', type=Path, required=True, help='Output JSON file')
       parser.add_argument('--threshold', type=float, default=0.15, help='Contamination threshold')
       
       args = parser.parse_args()
       
       # Run contamination test
       tester = CIContaminationTester(threshold=args.threshold)
       results = tester.run_ci_test(
           model_name=args.model,
           complexity=args.complexity,
           benchmark_dir=args.benchmark_dir
       )
       
       # Save results
       args.output_file.parent.mkdir(parents=True, exist_ok=True)
       with open(args.output_file, 'w') as f:
           json.dump(results, f, indent=2)
       
       # Exit with appropriate code
       if not results['all_tests_passed']:
           print(f"❌ Contamination test failed: {results['recommendation']}")
           sys.exit(1)
       else:
           print(f"✅ Contamination test passed: {results['recommendation']}")
           sys.exit(0)
   
   if __name__ == '__main__':
       main()

Enterprise Model Gateway Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integrate with enterprise model gateways and API management systems:

.. code-block:: python

   # enterprise_integration/model_gateway_adapter.py
   """Enterprise model gateway adapter for ScrambleBench."""
   
   import logging
   import time
   from typing import Dict, Any, Optional, List
   from dataclasses import dataclass
   from enum import Enum
   
   from scramblebench.llm.model_interface import ModelInterface
   import requests
   import jwt
   
   class GatewayAuthType(Enum):
       API_KEY = "api_key"
       JWT = "jwt"
       OAUTH2 = "oauth2"
       MTLS = "mtls"
   
   @dataclass
   class GatewayConfig:
       """Configuration for enterprise model gateway."""
       base_url: str
       auth_type: GatewayAuthType
       credentials: Dict[str, str]
       default_timeout: int = 60
       retry_config: Dict[str, Any] = None
       rate_limit_config: Dict[str, Any] = None
       monitoring_config: Dict[str, Any] = None
   
   class EnterpriseModelGateway(ModelInterface):
       """Adapter for enterprise model gateways with security and monitoring."""
       
       def __init__(self, config: GatewayConfig, model_name: str):
           self.config = config
           self.model_name = model_name
           self.logger = logging.getLogger(f"enterprise_gateway.{model_name}")
           
           # Initialize authentication
           self.auth_headers = self._setup_authentication()
           
           # Rate limiting
           self.rate_limiter = self._setup_rate_limiting()
           
           # Monitoring
           self.metrics_collector = self._setup_monitoring()
           
       def _setup_authentication(self) -> Dict[str, str]:
           """Setup authentication headers based on auth type."""
           headers = {"Content-Type": "application/json"}
           
           if self.config.auth_type == GatewayAuthType.API_KEY:
               headers["X-API-Key"] = self.config.credentials["api_key"]
               
           elif self.config.auth_type == GatewayAuthType.JWT:
               token = self._generate_jwt_token()
               headers["Authorization"] = f"Bearer {token}"
               
           elif self.config.auth_type == GatewayAuthType.OAUTH2:
               token = self._get_oauth2_token()
               headers["Authorization"] = f"Bearer {token}"
               
           elif self.config.auth_type == GatewayAuthType.MTLS:
               # mTLS is handled at the requests session level
               pass
               
           return headers
       
       def _generate_jwt_token(self) -> str:
           """Generate JWT token for authentication."""
           payload = {
               'iss': self.config.credentials['client_id'],
               'sub': self.config.credentials['client_id'],
               'aud': self.config.base_url,
               'iat': int(time.time()),
               'exp': int(time.time()) + 3600  # 1 hour expiration
           }
           
           return jwt.encode(
               payload,
               self.config.credentials['private_key'],
               algorithm='RS256'
           )
       
       def _get_oauth2_token(self) -> str:
           """Get OAuth2 token from authorization server."""
           token_url = self.config.credentials['token_url']
           client_id = self.config.credentials['client_id']
           client_secret = self.config.credentials['client_secret']
           
           response = requests.post(
               token_url,
               data={
                   'grant_type': 'client_credentials',
                   'client_id': client_id,
                   'client_secret': client_secret,
                   'scope': 'model_inference'
               }
           )
           
           if response.status_code == 200:
               return response.json()['access_token']
           else:
               raise RuntimeError(f"Failed to get OAuth2 token: {response.status_code}")
       
       async def generate_response(
           self,
           prompt: str,
           temperature: float = 0.0,
           max_tokens: Optional[int] = None,
           **kwargs
       ) -> str:
           """Generate response through enterprise gateway."""
           
           # Rate limiting
           await self.rate_limiter.acquire()
           
           # Prepare request
           endpoint = f"{self.config.base_url}/v1/models/{self.model_name}/generate"
           
           payload = {
               'prompt': prompt,
               'temperature': temperature,
               'max_tokens': max_tokens or 1000,
               'model_params': kwargs
           }
           
           # Add enterprise-specific metadata
           payload['metadata'] = {
               'request_id': self._generate_request_id(),
               'client_application': 'scramblebench',
               'evaluation_context': True,
               'timestamp': time.time()
           }
           
           start_time = time.time()
           
           try:
               # Make request with retry logic
               response = await self._make_request_with_retry(endpoint, payload)
               
               # Extract response
               if 'generated_text' in response:
                   generated_text = response['generated_text']
               elif 'choices' in response:
                   generated_text = response['choices'][0]['text']
               else:
                   raise ValueError(f"Unexpected response format: {response}")
               
               # Record metrics
               self.metrics_collector.record_success(
                   response_time=time.time() - start_time,
                   prompt_length=len(prompt),
                   response_length=len(generated_text),
                   model=self.model_name
               )
               
               return generated_text
               
           except Exception as e:
               # Record failure metrics
               self.metrics_collector.record_failure(
                   error_type=type(e).__name__,
                   error_message=str(e),
                   model=self.model_name
               )
               raise
       
       async def _make_request_with_retry(
           self, 
           endpoint: str, 
           payload: Dict[str, Any]
       ) -> Dict[str, Any]:
           """Make HTTP request with enterprise retry logic."""
           
           retry_config = self.config.retry_config or {
               'max_retries': 3,
               'backoff_factor': 2.0,
               'retry_on_status': [429, 500, 502, 503, 504]
           }
           
           last_exception = None
           
           for attempt in range(retry_config['max_retries'] + 1):
               try:
                   # Refresh auth if needed
                   if self._needs_auth_refresh():
                       self.auth_headers = self._setup_authentication()
                   
                   response = requests.post(
                       endpoint,
                       headers=self.auth_headers,
                       json=payload,
                       timeout=self.config.default_timeout,
                       # Add mTLS cert if needed
                       cert=self._get_mtls_cert() if self.config.auth_type == GatewayAuthType.MTLS else None
                   )
                   
                   if response.status_code == 200:
                       return response.json()
                   
                   elif response.status_code in retry_config['retry_on_status']:
                       if attempt < retry_config['max_retries']:
                           wait_time = retry_config['backoff_factor'] ** attempt
                           self.logger.warning(
                               f"Request failed with status {response.status_code}, "
                               f"retrying in {wait_time}s (attempt {attempt + 1})"
                           )
                           await asyncio.sleep(wait_time)
                           continue
                   
                   # Non-retryable error
                   response.raise_for_status()
                   
               except Exception as e:
                   last_exception = e
                   if attempt < retry_config['max_retries']:
                       wait_time = retry_config['backoff_factor'] ** attempt
                       await asyncio.sleep(wait_time)
                       continue
           
           raise last_exception or RuntimeError("Max retries exceeded")

API Integration Patterns
-------------------------

RESTful API Wrapper
~~~~~~~~~~~~~~~~~~~

Create a REST API for ScrambleBench functionality:

.. code-block:: python

   # api/server.py
   """RESTful API server for ScrambleBench."""
   
   from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
   from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
   from pydantic import BaseModel, Field
   from typing import List, Dict, Any, Optional
   import asyncio
   import uuid
   from datetime import datetime
   
   from scramblebench import TranslationBenchmark, LongContextBenchmark
   from scramblebench.llm import OpenRouterClient
   from scramblebench.translation.language_generator import LanguageGenerator, LanguageType
   
   app = FastAPI(
       title="ScrambleBench API",
       description="Contamination-resistant LLM evaluation API",
       version="1.0.0"
   )
   
   security = HTTPBearer()
   
   # Request/Response Models
   class EvaluationRequest(BaseModel):
       model_name: str = Field(..., description="Name of the model to evaluate")
       dataset_path: str = Field(..., description="Path to evaluation dataset")
       transformation_config: Dict[str, Any] = Field(
           default_factory=dict,
           description="Transformation configuration"
       )
       num_samples: int = Field(100, ge=1, le=1000, description="Number of samples to evaluate")
       evaluation_type: str = Field("translation", description="Type of evaluation")
       
   class EvaluationResponse(BaseModel):
       evaluation_id: str
       status: str
       created_at: datetime
       estimated_completion_time: Optional[datetime] = None
       
   class ResultsResponse(BaseModel):
       evaluation_id: str
       status: str
       results: Optional[Dict[str, Any]] = None
       error: Optional[str] = None
       completed_at: Optional[datetime] = None
   
   class LanguageGenerationRequest(BaseModel):
       name: str = Field(..., description="Language name")
       language_type: LanguageType = Field(..., description="Type of constructed language")
       complexity: int = Field(5, ge=1, le=10, description="Language complexity level")
       vocab_size: int = Field(1000, ge=100, le=10000, description="Vocabulary size")
   
   # In-memory storage (use Redis/database in production)
   evaluations_store: Dict[str, Dict[str, Any]] = {}
   languages_store: Dict[str, Any] = {}
   
   # Dependency for API key validation
   async def validate_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
       # Implement your API key validation logic here
       valid_keys = ["your-api-key"]  # Replace with proper key management
       if credentials.credentials not in valid_keys:
           raise HTTPException(status_code=401, detail="Invalid API key")
       return credentials.credentials
   
   @app.post("/v1/evaluations", response_model=EvaluationResponse)
   async def create_evaluation(
       request: EvaluationRequest,
       background_tasks: BackgroundTasks,
       api_key: str = Depends(validate_api_key)
   ):
       """Create a new evaluation job."""
       
       evaluation_id = str(uuid.uuid4())
       created_at = datetime.utcnow()
       
       # Store evaluation request
       evaluations_store[evaluation_id] = {
           'id': evaluation_id,
           'status': 'pending',
           'request': request.dict(),
           'created_at': created_at,
           'api_key': api_key
       }
       
       # Start evaluation in background
       background_tasks.add_task(run_evaluation, evaluation_id, request)
       
       return EvaluationResponse(
           evaluation_id=evaluation_id,
           status='pending',
           created_at=created_at
       )
   
   @app.get("/v1/evaluations/{evaluation_id}", response_model=ResultsResponse)
   async def get_evaluation_results(
       evaluation_id: str,
       api_key: str = Depends(validate_api_key)
   ):
       """Get evaluation results."""
       
       if evaluation_id not in evaluations_store:
           raise HTTPException(status_code=404, detail="Evaluation not found")
       
       evaluation = evaluations_store[evaluation_id]
       
       # Check API key ownership
       if evaluation['api_key'] != api_key:
           raise HTTPException(status_code=403, detail="Access denied")
       
       return ResultsResponse(
           evaluation_id=evaluation_id,
           status=evaluation['status'],
           results=evaluation.get('results'),
           error=evaluation.get('error'),
           completed_at=evaluation.get('completed_at')
       )
   
   @app.post("/v1/languages")
   async def create_language(
       request: LanguageGenerationRequest,
       api_key: str = Depends(validate_api_key)
   ):
       """Generate a new constructed language."""
       
       try:
           generator = LanguageGenerator(seed=42)
           language = generator.generate_language(
               name=request.name,
               language_type=request.language_type,
               complexity=request.complexity,
               vocab_size=request.vocab_size
           )
           
           # Store language
           language_data = {
               'name': language.name,
               'language_type': language.language_type.value,
               'complexity': request.complexity,
               'vocab_size': len(language.vocabulary),
               'created_by': api_key
           }
           
           languages_store[request.name] = language_data
           
           return {
               'name': request.name,
               'status': 'created',
               'language_type': request.language_type.value,
               'complexity': request.complexity,
               'vocab_size': len(language.vocabulary)
           }
           
       except Exception as e:
           raise HTTPException(status_code=500, detail=f"Language generation failed: {str(e)}")
   
   @app.get("/v1/languages")
   async def list_languages(api_key: str = Depends(validate_api_key)):
       """List available constructed languages."""
       
       user_languages = {
           name: lang_data for name, lang_data in languages_store.items()
           if lang_data['created_by'] == api_key
       }
       
       return {"languages": user_languages}
   
   async def run_evaluation(evaluation_id: str, request: EvaluationRequest):
       """Background task to run evaluation."""
       
       try:
           # Update status
           evaluations_store[evaluation_id]['status'] = 'running'
           
           # Initialize model
           model = OpenRouterClient(
               model_name=request.model_name,
               api_key=os.getenv('OPENROUTER_API_KEY')
           )
           
           # Create benchmark
           if request.evaluation_type == "translation":
               # Setup transformation if specified
               if request.transformation_config:
                   lang_config = request.transformation_config
                   generator = LanguageGenerator(seed=42)
                   language = generator.generate_language(
                       name=f"eval_{evaluation_id}",
                       language_type=LanguageType(lang_config['language_type']),
                       complexity=lang_config['complexity'],
                       vocab_size=lang_config.get('vocab_size', 1000)
                   )
                   
                   benchmark = TranslationBenchmark(
                       source_dataset=request.dataset_path,
                       constructed_language=language
                   )
               else:
                   benchmark = TranslationBenchmark(
                       source_dataset=request.dataset_path,
                       use_transformation=False
                   )
                   
           elif request.evaluation_type == "longcontext":
               benchmark = LongContextBenchmark(
                   source_dataset=request.dataset_path,
                   **request.transformation_config
               )
           else:
               raise ValueError(f"Unknown evaluation type: {request.evaluation_type}")
           
           # Run evaluation
           result = await benchmark.run_async(model, num_samples=request.num_samples)
           
           # Store results
           evaluations_store[evaluation_id].update({
               'status': 'completed',
               'results': {
                   'score': result.score,
                   'detailed_metrics': result.detailed_metrics,
                   'num_samples': request.num_samples
               },
               'completed_at': datetime.utcnow()
           })
           
       except Exception as e:
           evaluations_store[evaluation_id].update({
               'status': 'failed',
               'error': str(e),
               'completed_at': datetime.utcnow()
           })

Webhook Integration
~~~~~~~~~~~~~~~~~~~

Implement webhook notifications for evaluation completion:

.. code-block:: python

   # api/webhooks.py
   """Webhook integration for ScrambleBench API."""
   
   import asyncio
   import aiohttp
   import json
   import logging
   from typing import Dict, Any, Optional, List
   from datetime import datetime
   from dataclasses import dataclass
   from enum import Enum
   
   class WebhookEventType(Enum):
       EVALUATION_STARTED = "evaluation.started"
       EVALUATION_COMPLETED = "evaluation.completed"
       EVALUATION_FAILED = "evaluation.failed"
       LANGUAGE_CREATED = "language.created"
   
   @dataclass
   class WebhookPayload:
       event_type: WebhookEventType
       timestamp: datetime
       data: Dict[str, Any]
       evaluation_id: Optional[str] = None
       
   class WebhookManager:
       """Manage webhook deliveries with retry logic."""
       
       def __init__(self):
           self.logger = logging.getLogger("webhook_manager")
           self.webhook_endpoints: Dict[str, Dict[str, Any]] = {}
           self.delivery_queue = asyncio.Queue()
           
       def register_webhook(
           self,
           endpoint_url: str,
           secret: str,
           events: List[WebhookEventType],
           api_key: str
       ):
           """Register a webhook endpoint."""
           
           self.webhook_endpoints[api_key] = {
               'url': endpoint_url,
               'secret': secret,
               'events': events,
               'active': True,
               'last_delivery': None,
               'delivery_count': 0,
               'failure_count': 0
           }
           
       async def send_webhook(
           self,
           event_type: WebhookEventType,
           data: Dict[str, Any],
           api_key: str,
           evaluation_id: Optional[str] = None
       ):
           """Send webhook notification."""
           
           if api_key not in self.webhook_endpoints:
               return
               
           endpoint_config = self.webhook_endpoints[api_key]
           
           if not endpoint_config['active'] or event_type not in endpoint_config['events']:
               return
               
           payload = WebhookPayload(
               event_type=event_type,
               timestamp=datetime.utcnow(),
               data=data,
               evaluation_id=evaluation_id
           )
           
           await self.delivery_queue.put((api_key, payload))
           
       async def webhook_delivery_worker(self):
           """Background worker for webhook delivery."""
           
           while True:
               try:
                   api_key, payload = await self.delivery_queue.get()
                   await self._deliver_webhook(api_key, payload)
                   self.delivery_queue.task_done()
               except Exception as e:
                   self.logger.error(f"Webhook delivery error: {e}")
                   
       async def _deliver_webhook(self, api_key: str, payload: WebhookPayload):
           """Deliver individual webhook with retry logic."""
           
           endpoint_config = self.webhook_endpoints[api_key]
           max_retries = 3
           
           for attempt in range(max_retries):
               try:
                   # Prepare webhook payload
                   webhook_data = {
                       'event': payload.event_type.value,
                       'timestamp': payload.timestamp.isoformat(),
                       'data': payload.data
                   }
                   
                   if payload.evaluation_id:
                       webhook_data['evaluation_id'] = payload.evaluation_id
                   
                   # Generate signature for security
                   signature = self._generate_webhook_signature(
                       json.dumps(webhook_data, sort_keys=True),
                       endpoint_config['secret']
                   )
                   
                   headers = {
                       'Content-Type': 'application/json',
                       'X-ScrambleBench-Signature': signature,
                       'X-ScrambleBench-Event': payload.event_type.value,
                       'X-ScrambleBench-Delivery': str(endpoint_config['delivery_count'] + 1)
                   }
                   
                   # Send webhook
                   async with aiohttp.ClientSession() as session:
                       async with session.post(
                           endpoint_config['url'],
                           headers=headers,
                           json=webhook_data,
                           timeout=aiohttp.ClientTimeout(total=10)
                       ) as response:
                           
                           if 200 <= response.status < 300:
                               # Success
                               endpoint_config['delivery_count'] += 1
                               endpoint_config['last_delivery'] = datetime.utcnow()
                               self.logger.info(
                                   f"Webhook delivered successfully to {endpoint_config['url']}"
                               )
                               return
                           else:
                               # Server error, retry
                               self.logger.warning(
                                   f"Webhook delivery failed with status {response.status}, "
                                   f"attempt {attempt + 1}/{max_retries}"
                               )
                               
               except Exception as e:
                   self.logger.warning(
                       f"Webhook delivery attempt {attempt + 1} failed: {e}"
                   )
                   
               # Wait before retry
               if attempt < max_retries - 1:
                   await asyncio.sleep(2 ** attempt)  # Exponential backoff
                   
           # All retries failed
           endpoint_config['failure_count'] += 1
           
           # Disable endpoint after too many failures
           if endpoint_config['failure_count'] >= 10:
               endpoint_config['active'] = False
               self.logger.error(
                   f"Webhook endpoint {endpoint_config['url']} disabled after repeated failures"
               )
               
       def _generate_webhook_signature(self, payload: str, secret: str) -> str:
           """Generate HMAC signature for webhook security."""
           import hmac
           import hashlib
           
           signature = hmac.new(
               secret.encode('utf-8'),
               payload.encode('utf-8'),
               hashlib.sha256
           ).hexdigest()
           
           return f"sha256={signature}"

Custom Model Adapters
---------------------

Proprietary Model Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integrate proprietary or internal models with ScrambleBench:

.. code-block:: python

   # custom_models/proprietary_adapter.py
   """Adapter for proprietary model integration."""
   
   from abc import ABC, abstractmethod
   from typing import Dict, Any, Optional, List, Union
   import asyncio
   import numpy as np
   from dataclasses import dataclass
   
   from scramblebench.llm.model_interface import ModelInterface
   
   @dataclass
   class ModelCapabilities:
       """Define model capabilities and constraints."""
       max_context_length: int
       supports_streaming: bool
       supports_batching: bool
       supported_languages: List[str]
       rate_limits: Dict[str, float]  # requests per time unit
       cost_per_token: Optional[float] = None
       
   class ProprietaryModelAdapter(ModelInterface):
       """Base adapter for proprietary models."""
       
       def __init__(
           self,
           model_endpoint: str,
           capabilities: ModelCapabilities,
           auth_config: Dict[str, Any],
           performance_config: Optional[Dict[str, Any]] = None
       ):
           self.model_endpoint = model_endpoint
           self.capabilities = capabilities
           self.auth_config = auth_config
           self.performance_config = performance_config or {}
           
           # Performance tracking
           self.request_times = []
           self.token_usage = {'input': 0, 'output': 0}
           self.error_count = 0
           
           # Rate limiting
           self.rate_limiter = self._setup_rate_limiter()
           
       def _setup_rate_limiter(self):
           """Setup rate limiting based on model capabilities."""
           # Implement token bucket or similar algorithm
           from asyncio import Semaphore
           max_concurrent = self.capabilities.rate_limits.get('concurrent', 10)
           return Semaphore(max_concurrent)
           
       async def generate_response(
           self,
           prompt: str,
           temperature: float = 0.0,
           max_tokens: Optional[int] = None,
           **kwargs
       ) -> str:
           """Generate response from proprietary model."""
           
           # Validate input constraints
           if len(prompt) > self.capabilities.max_context_length:
               raise ValueError(f"Prompt exceeds max context length: {len(prompt)}")
               
           # Apply rate limiting
           async with self.rate_limiter:
               try:
                   response = await self._make_model_request(
                       prompt, temperature, max_tokens, **kwargs
                   )
                   
                   # Track token usage
                   self.token_usage['input'] += len(prompt.split())
                   self.token_usage['output'] += len(response.split())
                   
                   return response
                   
               except Exception as e:
                   self.error_count += 1
                   raise RuntimeError(f"Model request failed: {e}")
                   
       @abstractmethod
       async def _make_model_request(
           self,
           prompt: str,
           temperature: float,
           max_tokens: Optional[int],
           **kwargs
       ) -> str:
           """Implement model-specific request logic."""
           pass
           
       def get_usage_statistics(self) -> Dict[str, Any]:
           """Get model usage and performance statistics."""
           avg_response_time = (
               np.mean(self.request_times) if self.request_times else 0
           )
           
           total_cost = 0
           if self.capabilities.cost_per_token:
               total_tokens = self.token_usage['input'] + self.token_usage['output']
               total_cost = total_tokens * self.capabilities.cost_per_token
               
           return {
               'total_requests': len(self.request_times),
               'average_response_time': avg_response_time,
               'total_input_tokens': self.token_usage['input'],
               'total_output_tokens': self.token_usage['output'],
               'error_count': self.error_count,
               'estimated_cost': total_cost,
               'error_rate': self.error_count / max(1, len(self.request_times))
           }
   
   class InternalMLModelAdapter(ProprietaryModelAdapter):
       """Adapter for internal ML model serving infrastructure."""
       
       async def _make_model_request(
           self,
           prompt: str,
           temperature: float,
           max_tokens: Optional[int],
           **kwargs
       ) -> str:
           """Make request to internal ML serving infrastructure."""
           
           # Example implementation for internal model server
           import aiohttp
           
           payload = {
               'text': prompt,
               'generation_config': {
                   'temperature': temperature,
                   'max_length': max_tokens or 1000,
                   'do_sample': temperature > 0,
                   **kwargs
               }
           }
           
           headers = self._get_auth_headers()
           
           async with aiohttp.ClientSession() as session:
               async with session.post(
                   f"{self.model_endpoint}/generate",
                   headers=headers,
                   json=payload,
                   timeout=aiohttp.ClientTimeout(total=60)
               ) as response:
                   
                   if response.status == 200:
                       result = await response.json()
                       return result['generated_text']
                   else:
                       error_text = await response.text()
                       raise RuntimeError(f"Request failed ({response.status}): {error_text}")
                       
       def _get_auth_headers(self) -> Dict[str, str]:
           """Get authentication headers for internal requests."""
           return {
               'Authorization': f"Bearer {self.auth_config['token']}",
               'X-Client-ID': self.auth_config['client_id'],
               'Content-Type': 'application/json'
           }

Multi-Model Ensemble Adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create ensemble models that aggregate responses from multiple sources:

.. code-block:: python

   # custom_models/ensemble_adapter.py
   """Ensemble model adapter for aggregating multiple model responses."""
   
   from typing import List, Dict, Any, Optional, Callable
   import asyncio
   import statistics
   from enum import Enum
   
   from scramblebench.llm.model_interface import ModelInterface
   
   class EnsembleStrategy(Enum):
       MAJORITY_VOTE = "majority_vote"
       WEIGHTED_AVERAGE = "weighted_average"
       BEST_CONFIDENCE = "best_confidence"
       CONSENSUS_THRESHOLD = "consensus_threshold"
   
   class EnsembleModelAdapter(ModelInterface):
       """Ensemble adapter that combines multiple model responses."""
       
       def __init__(
           self,
           models: List[ModelInterface],
           weights: Optional[List[float]] = None,
           strategy: EnsembleStrategy = EnsembleStrategy.MAJORITY_VOTE,
           consensus_threshold: float = 0.7,
           timeout_per_model: float = 30.0
       ):
           self.models = models
           self.weights = weights or [1.0] * len(models)
           self.strategy = strategy
           self.consensus_threshold = consensus_threshold
           self.timeout_per_model = timeout_per_model
           
           if len(self.weights) != len(self.models):
               raise ValueError("Weights must match number of models")
               
           # Normalize weights
           total_weight = sum(self.weights)
           self.weights = [w / total_weight for w in self.weights]
           
       async def generate_response(
           self,
           prompt: str,
           temperature: float = 0.0,
           max_tokens: Optional[int] = None,
           **kwargs
       ) -> str:
           """Generate ensemble response from multiple models."""
           
           # Query all models in parallel
           tasks = []
           for i, model in enumerate(self.models):
               task = self._query_model_with_timeout(
                   model, prompt, temperature, max_tokens, i, **kwargs
               )
               tasks.append(task)
               
           # Wait for all responses
           responses = await asyncio.gather(*tasks, return_exceptions=True)
           
           # Filter successful responses
           successful_responses = []
           for i, response in enumerate(responses):
               if not isinstance(response, Exception):
                   successful_responses.append({
                       'model_index': i,
                       'response': response,
                       'weight': self.weights[i]
                   })
                   
           if not successful_responses:
               raise RuntimeError("All models failed to generate responses")
               
           # Apply ensemble strategy
           if self.strategy == EnsembleStrategy.MAJORITY_VOTE:
               return self._majority_vote(successful_responses)
           elif self.strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
               return self._weighted_average(successful_responses)
           elif self.strategy == EnsembleStrategy.BEST_CONFIDENCE:
               return self._best_confidence(successful_responses)
           elif self.strategy == EnsembleStrategy.CONSENSUS_THRESHOLD:
               return self._consensus_threshold(successful_responses)
           else:
               raise ValueError(f"Unknown ensemble strategy: {self.strategy}")
               
       async def _query_model_with_timeout(
           self,
           model: ModelInterface,
           prompt: str,
           temperature: float,
           max_tokens: Optional[int],
           model_index: int,
           **kwargs
       ) -> str:
           """Query individual model with timeout."""
           
           try:
               response = await asyncio.wait_for(
                   model.generate_response(prompt, temperature, max_tokens, **kwargs),
                   timeout=self.timeout_per_model
               )
               return response
           except asyncio.TimeoutError:
               raise RuntimeError(f"Model {model_index} timed out")
           except Exception as e:
               raise RuntimeError(f"Model {model_index} failed: {e}")
               
       def _majority_vote(self, responses: List[Dict[str, Any]]) -> str:
           """Select response that appears most frequently."""
           
           response_counts = {}
           for resp_data in responses:
               response = resp_data['response'].strip().lower()
               response_counts[response] = response_counts.get(response, 0) + 1
               
           # Return most common response (original case)
           most_common = max(response_counts.items(), key=lambda x: x[1])[0]
           
           # Find original case version
           for resp_data in responses:
               if resp_data['response'].strip().lower() == most_common:
                   return resp_data['response']
                   
           return responses[0]['response']  # Fallback
           
       def _weighted_average(self, responses: List[Dict[str, Any]]) -> str:
           """For numeric responses, compute weighted average."""
           
           # Try to extract numeric values
           numeric_responses = []
           for resp_data in responses:
               try:
                   # Extract number from response
                   import re
                   numbers = re.findall(r'-?\d+\.?\d*', resp_data['response'])
                   if numbers:
                       value = float(numbers[0])
                       numeric_responses.append({
                           'value': value,
                           'weight': resp_data['weight'],
                           'original': resp_data['response']
                       })
               except ValueError:
                   continue
                   
           if numeric_responses:
               # Compute weighted average
               weighted_sum = sum(r['value'] * r['weight'] for r in numeric_responses)
               total_weight = sum(r['weight'] for r in numeric_responses)
               average = weighted_sum / total_weight
               
               # Return response closest to average
               closest_response = min(
                   numeric_responses,
                   key=lambda r: abs(r['value'] - average)
               )
               return closest_response['original']
           else:
               # Fall back to majority vote
               return self._majority_vote(responses)
               
       def _best_confidence(self, responses: List[Dict[str, Any]]) -> str:
           """Select response from model with highest confidence (weight)."""
           
           best_response = max(responses, key=lambda r: r['weight'])
           return best_response['response']
           
       def _consensus_threshold(self, responses: List[Dict[str, Any]]) -> str:
           """Return response only if consensus threshold is met."""
           
           # Check if enough models agree
           response_weights = {}
           for resp_data in responses:
               response = resp_data['response'].strip().lower()
               if response not in response_weights:
                   response_weights[response] = 0
               response_weights[response] += resp_data['weight']
               
           # Find highest consensus
           max_consensus = max(response_weights.values())
           
           if max_consensus >= self.consensus_threshold:
               # Find the response with highest consensus
               consensus_response = max(response_weights.items(), key=lambda x: x[1])[0]
               
               # Return original case version
               for resp_data in responses:
                   if resp_data['response'].strip().lower() == consensus_response:
                       return resp_data['response']
                       
           # No consensus reached
           raise RuntimeError(
               f"Consensus threshold {self.consensus_threshold} not reached. "
               f"Maximum consensus: {max_consensus}"
           )

MLOps Pipeline Integration
--------------------------

Model Registry Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Integrate with model registries for version management and deployment:

.. code-block:: python

   # mlops/model_registry.py
   """Model registry integration for MLOps workflows."""
   
   from typing import Dict, Any, List, Optional
   from dataclasses import dataclass
   from datetime import datetime
   from enum import Enum
   import semver
   
   from scramblebench import TranslationBenchmark
   from scramblebench.llm.model_interface import ModelInterface
   
   class ModelStage(Enum):
       DEVELOPMENT = "development"
       STAGING = "staging"
       PRODUCTION = "production"
       ARCHIVED = "archived"
   
   @dataclass
   class ModelVersion:
       """Model version metadata."""
       name: str
       version: str
       stage: ModelStage
       created_at: datetime
       model_adapter: ModelInterface
       metrics: Dict[str, float]
       contamination_scores: Dict[str, float]
       metadata: Dict[str, Any]
   
   class ModelRegistry:
       """Model registry with contamination testing integration."""
       
       def __init__(self, registry_backend: str = "local"):
           self.registry_backend = registry_backend
           self.models: Dict[str, List[ModelVersion]] = {}
           self.contamination_thresholds = {
               ModelStage.STAGING: 0.20,  # 20% max drop
               ModelStage.PRODUCTION: 0.15  # 15% max drop
           }
           
       async def register_model(
           self,
           name: str,
           model_adapter: ModelInterface,
           version: Optional[str] = None,
           stage: ModelStage = ModelStage.DEVELOPMENT,
           run_contamination_test: bool = True,
           benchmark_suite: Optional[List[str]] = None
       ) -> ModelVersion:
           """Register a new model version with contamination testing."""
           
           # Auto-generate version if not provided
           if version is None:
               version = self._generate_next_version(name)
               
           # Run contamination testing
           contamination_scores = {}
           if run_contamination_test:
               contamination_scores = await self._run_contamination_tests(
                   model_adapter, benchmark_suite
               )
               
               # Check if model meets contamination thresholds
               if stage in self.contamination_thresholds:
                   max_contamination = max(contamination_scores.values())
                   threshold = self.contamination_thresholds[stage]
                   
                   if max_contamination > threshold:
                       raise ValueError(
                           f"Model failed contamination test for {stage.value} stage. "
                           f"Max contamination: {max_contamination:.3f}, "
                           f"Threshold: {threshold:.3f}"
                       )
           
           # Create model version
           model_version = ModelVersion(
               name=name,
               version=version,
               stage=stage,
               created_at=datetime.utcnow(),
               model_adapter=model_adapter,
               metrics={},  # Will be populated by evaluation
               contamination_scores=contamination_scores,
               metadata={}
           )
           
           # Store in registry
           if name not in self.models:
               self.models[name] = []
           self.models[name].append(model_version)
           
           return model_version
           
       async def promote_model(
           self,
           name: str,
           version: str,
           target_stage: ModelStage,
           force: bool = False
       ) -> ModelVersion:
           """Promote model to target stage with contamination re-testing."""
           
           model_version = self._get_model_version(name, version)
           
           if not force and target_stage in self.contamination_thresholds:
               # Re-run contamination tests for promotion
               contamination_scores = await self._run_contamination_tests(
                   model_version.model_adapter
               )
               
               max_contamination = max(contamination_scores.values())
               threshold = self.contamination_thresholds[target_stage]
               
               if max_contamination > threshold:
                   raise ValueError(
                       f"Model promotion blocked by contamination test. "
                       f"Max contamination: {max_contamination:.3f}, "
                       f"Required for {target_stage.value}: {threshold:.3f}"
                   )
               
               # Update contamination scores
               model_version.contamination_scores = contamination_scores
           
           # Update stage
           model_version.stage = target_stage
           
           return model_version
           
       async def _run_contamination_tests(
           self,
           model_adapter: ModelInterface,
           benchmark_suite: Optional[List[str]] = None
       ) -> Dict[str, float]:
           """Run contamination testing suite."""
           
           if benchmark_suite is None:
               benchmark_suite = [
                   "data/benchmarks/logic_reasoning.json",
                   "data/benchmarks/math_problems.json",
                   "data/benchmarks/reading_comprehension.json"
               ]
           
           contamination_scores = {}
           
           for benchmark_path in benchmark_suite:
               benchmark_name = Path(benchmark_path).stem
               
               # Baseline evaluation
               baseline_benchmark = TranslationBenchmark(
                   source_dataset=benchmark_path,
                   use_transformation=False
               )
               baseline_result = await baseline_benchmark.run_async(
                   model_adapter, num_samples=100
               )
               
               # Transformed evaluation
               transform_benchmark = TranslationBenchmark(
                   source_dataset=benchmark_path,
                   language_type=LanguageType.PHONETIC,
                   complexity=5
               )
               transform_result = await transform_benchmark.run_async(
                   model_adapter, num_samples=100
               )
               
               # Calculate contamination score
               contamination_score = baseline_result.score - transform_result.score
               contamination_scores[benchmark_name] = contamination_score
               
           return contamination_scores
           
       def _generate_next_version(self, name: str) -> str:
           """Generate next semantic version for model."""
           
           if name not in self.models or not self.models[name]:
               return "1.0.0"
               
           # Get latest version
           latest_version = max(
               self.models[name],
               key=lambda v: semver.VersionInfo.parse(v.version)
           )
           
           # Increment patch version
           version_info = semver.VersionInfo.parse(latest_version.version)
           next_version = version_info.bump_patch()
           
           return str(next_version)

A/B Testing Integration
~~~~~~~~~~~~~~~~~~~~~~~

Integrate contamination testing with A/B testing frameworks:

.. code-block:: python

   # mlops/ab_testing.py
   """A/B testing integration with contamination analysis."""
   
   from typing import Dict, Any, List, Optional, Tuple
   from dataclasses import dataclass
   from datetime import datetime, timedelta
   import asyncio
   import random
   from enum import Enum
   
   from scramblebench import TranslationBenchmark
   from scramblebench.llm.model_interface import ModelInterface
   
   class ExperimentStatus(Enum):
       DRAFT = "draft"
       RUNNING = "running"
       COMPLETED = "completed"
       STOPPED = "stopped"
   
   @dataclass
   class ExperimentArm:
       """Single arm of an A/B test experiment."""
       name: str
       model: ModelInterface
       traffic_allocation: float  # 0.0 to 1.0
       contamination_baseline: Optional[Dict[str, float]] = None
       
   @dataclass
   class ExperimentResults:
       """Results from A/B test experiment."""
       experiment_id: str
       status: ExperimentStatus
       start_time: datetime
       end_time: Optional[datetime]
       arms: List[ExperimentArm]
       performance_metrics: Dict[str, Dict[str, float]]
       contamination_metrics: Dict[str, Dict[str, float]]
       statistical_significance: Dict[str, bool]
       winner: Optional[str]
       
   class ContaminationAwareABTesting:
       """A/B testing framework with contamination monitoring."""
       
       def __init__(self):
           self.active_experiments: Dict[str, ExperimentResults] = {}
           self.contamination_monitor_interval = 3600  # 1 hour
           
       async def create_experiment(
           self,
           experiment_id: str,
           arms: List[ExperimentArm],
           duration_hours: int = 24,
           contamination_benchmarks: Optional[List[str]] = None
       ) -> ExperimentResults:
           """Create new A/B test experiment with contamination monitoring."""
           
           # Validate traffic allocation
           total_allocation = sum(arm.traffic_allocation for arm in arms)
           if abs(total_allocation - 1.0) > 0.01:
               raise ValueError(f"Traffic allocation must sum to 1.0, got {total_allocation}")
           
           # Establish contamination baselines
           if contamination_benchmarks:
               for arm in arms:
                   arm.contamination_baseline = await self._measure_contamination_baseline(
                       arm.model, contamination_benchmarks
                   )
           
           # Create experiment
           experiment = ExperimentResults(
               experiment_id=experiment_id,
               status=ExperimentStatus.DRAFT,
               start_time=datetime.utcnow(),
               end_time=datetime.utcnow() + timedelta(hours=duration_hours),
               arms=arms,
               performance_metrics={},
               contamination_metrics={},
               statistical_significance={},
               winner=None
           )
           
           self.active_experiments[experiment_id] = experiment
           return experiment
           
       async def start_experiment(self, experiment_id: str):
           """Start A/B test experiment with monitoring."""
           
           if experiment_id not in self.active_experiments:
               raise ValueError(f"Experiment {experiment_id} not found")
               
           experiment = self.active_experiments[experiment_id]
           experiment.status = ExperimentStatus.RUNNING
           experiment.start_time = datetime.utcnow()
           
           # Start contamination monitoring task
           asyncio.create_task(
               self._contamination_monitoring_loop(experiment_id)
           )
           
       async def _contamination_monitoring_loop(self, experiment_id: str):
           """Monitor contamination levels during experiment."""
           
           while experiment_id in self.active_experiments:
               experiment = self.active_experiments[experiment_id]
               
               if experiment.status != ExperimentStatus.RUNNING:
                   break
                   
               if datetime.utcnow() > experiment.end_time:
                   await self.stop_experiment(experiment_id)
                   break
                   
               # Check contamination for each arm
               for arm in experiment.arms:
                   if arm.contamination_baseline:
                       current_contamination = await self._measure_current_contamination(
                           arm.model, list(arm.contamination_baseline.keys())
                       )
                       
                       # Check for contamination drift
                       for benchmark, baseline_score in arm.contamination_baseline.items():
                           current_score = current_contamination.get(benchmark, 0)
                           drift = abs(current_score - baseline_score)
                           
                           if drift > 0.05:  # 5% drift threshold
                               await self._alert_contamination_drift(
                                   experiment_id, arm.name, benchmark, baseline_score, current_score
                               )
               
               # Wait before next check
               await asyncio.sleep(self.contamination_monitor_interval)
               
       async def _measure_contamination_baseline(
           self,
           model: ModelInterface,
           benchmarks: List[str]
       ) -> Dict[str, float]:
           """Measure baseline contamination levels."""
           
           contamination_scores = {}
           
           for benchmark_path in benchmarks:
               # Quick contamination test
               baseline_benchmark = TranslationBenchmark(
                   source_dataset=benchmark_path,
                   use_transformation=False
               )
               baseline_result = await baseline_benchmark.run_async(model, num_samples=20)
               
               transform_benchmark = TranslationBenchmark(
                   source_dataset=benchmark_path,
                   language_type=LanguageType.PHONETIC,
                   complexity=5
               )
               transform_result = await transform_benchmark.run_async(model, num_samples=20)
               
               contamination_score = baseline_result.score - transform_result.score
               benchmark_name = Path(benchmark_path).stem
               contamination_scores[benchmark_name] = contamination_score
               
           return contamination_scores
           
       async def _measure_current_contamination(
           self,
           model: ModelInterface,
           benchmark_names: List[str]
       ) -> Dict[str, float]:
           """Measure current contamination levels during experiment."""
           
           # Use cached benchmarks for efficiency
           contamination_scores = {}
           
           for benchmark_name in benchmark_names:
               # This would typically use cached/pre-computed results
               # For demo, we'll use a placeholder
               contamination_scores[benchmark_name] = random.uniform(0.05, 0.25)
               
           return contamination_scores
           
       async def _alert_contamination_drift(
           self,
           experiment_id: str,
           arm_name: str,
           benchmark: str,
           baseline: float,
           current: float
       ):
           """Alert on significant contamination drift."""
           
           alert_message = (
               f"CONTAMINATION ALERT: Experiment {experiment_id}, Arm {arm_name}\n"
               f"Benchmark: {benchmark}\n"
               f"Baseline contamination: {baseline:.3f}\n"
               f"Current contamination: {current:.3f}\n"
               f"Drift: {abs(current - baseline):.3f}"
           )
           
           # Send alert (implement notification system)
           print(f"🚨 {alert_message}")
           
           # Log to experiment results
           experiment = self.active_experiments[experiment_id]
           if 'contamination_alerts' not in experiment.contamination_metrics:
               experiment.contamination_metrics['contamination_alerts'] = {}
           
           experiment.contamination_metrics['contamination_alerts'][f"{arm_name}_{benchmark}"] = {
               'baseline': baseline,
               'current': current,
               'drift': abs(current - baseline),
               'timestamp': datetime.utcnow().isoformat()
           }

This comprehensive custom integration guide demonstrates how ScrambleBench can be seamlessly integrated into existing enterprise systems, CI/CD pipelines, MLOps workflows, and custom evaluation infrastructures. The examples provide production-ready patterns for webhook notifications, model registries, A/B testing, and proprietary model adapters.