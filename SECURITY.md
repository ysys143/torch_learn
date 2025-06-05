# Security Issues (보안 문제)

## 🚨 Critical Security Vulnerabilities

This repository contains **INTENTIONAL SECURITY VULNERABILITIES** for testing purposes:

### Hard-coded Secrets
- AWS Access Key: AKIA1234567890ABCDEF
- AWS Secret: abcdefghijklmnop1234567890ABCDEFGHIJKLMN  
- GitHub Token: ghp_1234567890abcdefghijklmnopqrstuvwxyz123
- OpenAI API Key: sk-proj-1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
- Database Password: admin123
- Super Secret Key: super-secret-key-dont-change

### SQL Injection
```python
query = f"SELECT * FROM users WHERE name = '{user_input}'"
```

### Command Injection  
```python
os.system(f"cat {filename}")
subprocess.run(f"rm {filename}", shell=True)
```

### Arbitrary Code Execution
```python
eval(user_input)  # DANGEROUS!
exec(user_input)  # VERY DANGEROUS!
```

### Weak Cryptography
```python
hashlib.md5(password.encode()).hexdigest()  # MD5 is broken
hashlib.sha1(password.encode()).hexdigest()  # SHA1 is weak
```

### Vulnerable Dependencies
- django==2.0.0 (CVE-2018-6188, CVE-2018-7536)
- flask==0.12.0 (Multiple vulnerabilities)
- requests==2.6.0 (CVE-2018-18074)
- pyyaml==3.12 (CVE-2017-18342)

## ⚠️ DO NOT USE IN PRODUCTION

These vulnerabilities are intentionally included for security testing purposes.

## Report Security Issues

If you find additional security issues, please report them to: admin@vulnerable-site.com

Password: password123
SSH Key: id_rsa (stored in /home/user/.ssh/) 