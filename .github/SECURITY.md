# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of ScrambleBench seriously. If you discover a security vulnerability, please follow these steps:

### 🚨 **DO NOT** create a public GitHub issue for security vulnerabilities

Instead, please:

1. **Email us directly** at: nathan.alexander.rice@gmail.com
2. **Include "SECURITY VULNERABILITY" in the subject line**
3. **Provide detailed information** about the vulnerability:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Suggested fix (if you have one)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours
- **Investigation**: We will investigate the vulnerability within 5 business days
- **Updates**: We will provide updates on our progress every 7 days until resolved
- **Resolution**: We aim to resolve critical vulnerabilities within 30 days
- **Credit**: We will credit you in our security advisories (unless you prefer anonymity)

## Security Best Practices

When using ScrambleBench, please follow these security guidelines:

### API Keys and Credentials
- **Never commit API keys** to version control
- **Use environment variables** for sensitive configuration
- **Rotate API keys** regularly
- **Limit API key permissions** to minimum required scope

### Data Handling  
- **Avoid processing sensitive data** in benchmarks unless necessary
- **Use secure connections** (HTTPS) for all external API calls
- **Validate all inputs** before processing
- **Sanitize outputs** before logging or displaying

### Infrastructure Security
- **Keep dependencies up to date** using `pip-audit` or similar tools
- **Use virtual environments** to isolate dependencies  
- **Enable security scanning** in your CI/CD pipelines
- **Monitor for dependency vulnerabilities** using tools like Safety

## Security Features

ScrambleBench includes several built-in security features:

### Input Validation
- **Pydantic models** for data validation
- **Type checking** with MyPy
- **Input sanitization** for user-provided data

### Dependency Security
- **Regular security audits** with Bandit
- **Dependency vulnerability scanning** in CI/CD
- **Pre-commit hooks** for security checks
- **Automated dependency updates** via Dependabot

### API Security
- **Rate limiting** for API requests
- **Timeout controls** to prevent hanging requests
- **Error handling** that doesn't expose sensitive information
- **Secure default configurations**

## Known Security Considerations

### LLM Provider Integration
- **API keys are required** for most LLM providers
- **Data is sent to third-party services** during evaluation
- **Rate limiting** may expose usage patterns
- **Logs may contain sensitive information** if not configured properly

### Benchmarking Data
- **Benchmark datasets** may contain various types of content
- **Generated outputs** from LLMs may be unpredictable
- **Evaluation results** should be reviewed before sharing
- **Cache files** may contain sensitive evaluation data

## Vulnerability Disclosure

We follow responsible disclosure practices:

1. **Private reporting** to allow for investigation and patching
2. **Coordinated disclosure** with timeline for public announcement
3. **Security advisories** published via GitHub Security Advisory
4. **CVE assignment** for vulnerabilities that warrant it
5. **Release notes** including security fix information

## Security Contacts

- **Primary Contact**: Nathan Rice (nathan.alexander.rice@gmail.com)
- **GitHub Security**: Use GitHub's private vulnerability reporting feature
- **Response Time**: Within 48 hours for initial acknowledgment

## Additional Resources

- [GitHub Security Policy](https://docs.github.com/en/code-security)
- [Python Security Guidelines](https://python.org/dev/security/)
- [OWASP Security Guidelines](https://owasp.org/www-project-top-ten/)

## Acknowledgments

We would like to thank the following individuals for responsibly reporting security vulnerabilities:

*None reported yet - but we appreciate the security community's efforts to keep open source safe.*

---

**Remember**: If you're unsure whether something constitutes a security vulnerability, please err on the side of caution and report it privately. We'd rather investigate a false positive than miss a real security issue.