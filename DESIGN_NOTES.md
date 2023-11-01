# A version w/o stable indexes?

We can have a version w/o stable indexes but using
"input" and "simplified" tables if we buffer births
according to:

1. A hash of node -> Vec<Edge>
2. We keep a stack of those vectors after we 
   use them, allowing us to avoid constant reallocation.
