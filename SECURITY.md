# Security policy

kernelforge generates GPU and C source code from graphs you construct and runs it locally. It does
not open network connections. The C oracle compiles and loads generated code into the Python process,
so only compile graphs you trust.

Please report vulnerabilities privately through GitHub's "Report a vulnerability" (Security tab) on
https://github.com/coopersimpson101-debug/kernelforge rather than in a public issue.
