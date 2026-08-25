# Teacher bake-off

Date: 2026-08-25  ·  200 examples from the **val** split (test remains sealed).

| model | micro-F1 | P | R | schema-invalid | hallucinated | cost | F1 per $ |
|---|---|---|---|---|---|---|---|
| `accounts/fireworks/models/qwen3p7-plus` | **0.832** | 0.847 | 0.818 | 0.0% | 0 | $0.0942 | 9 |

## accounts/fireworks/models/qwen3p7-plus

```
label                      P       R      F1      n
DATE                   0.990   0.976   0.983    209
GIVENNAME              0.677   0.683   0.680    199
SURNAME                0.670   0.702   0.686    171
CITY                   0.826   0.820   0.823    133
EMAIL                  1.000   1.000   1.000    131
TELEPHONENUM           0.911   0.981   0.944    104
STREET                 0.885   0.944   0.914     90
ZIPCODE                0.782   0.847   0.813     72
IDCARDNUM              0.812   0.245   0.377     53
CREDITCARDNUMBER       1.000   0.872   0.932     47
TAXNUM                 1.000   0.718   0.836     39
SOCIALNUM              1.000   0.571   0.727     35
--------------------------------------------------
MICRO                  0.847   0.818   0.832   1283

examples: 200  schema-invalid: 0 (0.0%)  hallucinated values: 0
```

**Total spend: $0.0942**
