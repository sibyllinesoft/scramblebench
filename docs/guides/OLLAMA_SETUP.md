# Ollama Integration Setup Guide

This guide helps you set up Ollama integration for ScrambleBench to run preliminary benchmarks with small, local models.

## Quick Start

### 1. Install Ollama

**Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [ollama.ai](https://ollama.ai/download)

### 2. Start Ollama Server
```bash
ollama serve
```
*Keep this running in a separate terminal*

### 3. Pull Small Test Models
```bash
# Ultra-fast model for development (1.3GB)
ollama pull llama3.2:1b

# Balanced model for better results (2.3GB)
ollama pull phi3:3.8b

# Alternative efficient model (1.6GB)  
ollama pull gemma2:2b
```

### 4. Test Integration
```bash
cd /home/nathan/Projects/scramblebench
python run_ollama_test.py
```

### 5. Run Quick Test
```bash
python -m scramblebench.cli evaluate configs/evaluation/ollama_quick_test.yaml
```

## Available Models

### Ultra-Fast Models (< 2GB)
Perfect for development and quick testing:

| Model | Size | Parameters | Use Case |
|-------|------|------------|----------|
| `llama3.2:1b` | 1.3GB | 1B | Ultra-fast development testing |
| `gemma2:2b` | 1.6GB | 2B | Efficient general tasks |
| `qwen2:0.5b` | 0.7GB | 0.5B | Minimal resource testing |

### Balanced Models (2-4GB)  
Good quality for preliminary benchmarks:

| Model | Size | Parameters | Use Case |
|-------|------|------------|----------|
| `phi3:3.8b` | 2.3GB | 3.8B | Recommended for development |
| `llama3.2:3b` | 2.0GB | 3B | Balanced speed/quality |
| `gemma2:7b` | 4.3GB | 7B | Higher quality results |

### Quality Models (4-8GB)
For comprehensive evaluation:

| Model | Size | Parameters | Use Case |
|-------|------|------------|----------|
| `phi3:14b` | 7.9GB | 14B | High-quality reasoning |
| `llama3.1:8b` | 4.7GB | 8B | General-purpose excellence |
| `gemma2:9b` | 5.4GB | 9B | Strong performance |

## Configuration Files

### Quick Test (5 samples, ~2 minutes)
```bash
python -m scramblebench.cli evaluate configs/evaluation/ollama_quick_test.yaml
```

### Preliminary Benchmarks (20 samples, ~10 minutes)
```bash
python -m scramblebench.cli evaluate configs/evaluation/ollama_preliminary.yaml
```

## Troubleshooting

### Common Issues

**1. "Cannot connect to Ollama server"**
- Make sure Ollama is running: `ollama serve`
- Check if port 11434 is available: `curl http://localhost:11434/api/tags`
- Restart Ollama if needed

**2. "Model not found"**  
- Pull the required model: `ollama pull llama3.2:1b`
- Check available models: `ollama list`
- Verify model name spelling in config

**3. "Request timeout"**
- First run may be slow due to model loading
- Increase timeout in config file
- Try smaller model for testing

**4. "Memory issues"**
- Check available RAM: models need 2-4GB+ free memory
- Close other applications
- Use smaller model (e.g., `llama3.2:1b`)

### Performance Tips

**For Limited RAM (< 8GB):**
- Use `llama3.2:1b` or `gemma2:2b`
- Process samples sequentially (already default)
- Close other applications

**For Better Performance:**
- Use models with GPU acceleration if available
- Ensure adequate cooling for sustained inference
- Use SSD storage for faster model loading

**For Development:**
- Start with `ollama_quick_test.yaml` (5 samples)
- Use smallest model first (`llama3.2:1b`)
- Gradually increase sample size and model complexity

## Integration Details

### Model Interface
All Ollama models implement the standard ScrambleBench `ModelInterface`:
```python
from scramblebench.llm import OllamaClient, ModelConfig

# Create client
client = OllamaClient("phi3:3.8b", config=ModelConfig(temperature=0.0))
client.initialize()

# Generate text
response = client.generate("What is machine learning?")
print(response.text)
```

### Factory Pattern
Use the model factory for dynamic creation:
```python
from scramblebench.llm import ModelFactory

model = ModelFactory.create_model(
    provider="ollama",
    model_name="gemma2:2b",
    config=ModelConfig(max_tokens=512)
)
```

### Evaluation Pipeline
Ollama models integrate seamlessly with the evaluation pipeline:
- Sequential processing optimized for local inference
- Automatic error handling and retry logic
- Performance monitoring and statistics
- Progress tracking with detailed logging

## Next Steps

1. **Verify Setup:** Run `python run_ollama_test.py`
2. **Quick Test:** Run `configs/evaluation/ollama_quick_test.yaml`  
3. **Preliminary Benchmarks:** Run `configs/evaluation/ollama_preliminary.yaml`
4. **Custom Configs:** Create your own evaluation configurations
5. **Scale Up:** Try larger models and more comprehensive benchmarks

## Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Available Models](https://ollama.ai/models)
- [ScrambleBench Documentation](./docs/)
- [Model Performance Comparison](./docs/model_comparison.md)

## Support

For issues:
1. Check this guide's troubleshooting section
2. Verify Ollama installation: `ollama --version`
3. Test basic functionality: `ollama run llama3.2:1b "Hello"`
4. Check ScrambleBench logs for detailed error information