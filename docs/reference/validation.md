# Validation

`validate_tessellation(level='strict')` raises exactly when the completed
`TessellationDiagnostics.ok` is false. `require_reciprocity=False` still
inspects periodic reciprocity but reports missing shifts as info and
orphan/mismatched faces as warnings. Normalized-topology strict validation also
uses overall error severity, so warning-only Euler/incidence findings remain
nonfatal even when their descriptive subcheck is false.

::: pyvoro2.validation
:::
