# Runtime and paradigm modularity

The stable scientific boundary is an operator/readout pair:

- a state is a tree of arrays;
- a transition maps one state to the next;
- a readout maps a state to one declared signal channel;
- metadata records semantics, geometry, approximation family, and claim level.

The current implementation uses JAX pytrees and JAX transformations. A future runtime adapter for
Modular/MAX should preserve this boundary rather than reproduce the scientific logic in a
second registry. Such an adapter needs array conversion, deterministic dtype/shape contracts, and
agreement tests against JAX reference outputs. Runtime kernel performance and scientific equivalence
are separate questions.


