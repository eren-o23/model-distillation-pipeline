# Teacher bake-off

Date: 2026-08-25  ·  200 examples from the **val** split (test remains sealed).

| model | micro-F1 | P | R | schema-invalid | hallucinated | cost | F1 per $ |
|---|---|---|---|---|---|---|---|
| `accounts/fireworks/models/qwen3p7-plus` | **0.828** | 0.806 | 0.852 | 0.0% | 1 | $0.0874 | 9 |
| `accounts/fireworks/models/deepseek-v4-pro` | **0.833** | 0.818 | 0.849 | 0.0% | 1 | $0.3001 | 3 |

## accounts/fireworks/models/qwen3p7-plus

```
label                      P       R      F1      n
DATE                   0.990   0.990   0.990    209
GIVENNAME              0.687   0.693   0.690    199
SURNAME                0.680   0.708   0.693    171
CITY                   0.800   0.812   0.806    133
EMAIL                  1.000   1.000   1.000    131
TELEPHONENUM           0.866   0.990   0.924    104
STREET                 0.857   0.933   0.894     90
ZIPCODE                0.779   0.833   0.805     72
IDCARDNUM              0.423   0.887   0.573     53
CREDITCARDNUMBER       0.958   0.979   0.968     47
TAXNUM                 0.967   0.744   0.841     39
SOCIALNUM              1.000   0.543   0.704     35
--------------------------------------------------
MICRO                  0.806   0.852   0.828   1283

examples: 200  schema-invalid: 0 (0.0%)  hallucinated values: 1
```

## accounts/fireworks/models/deepseek-v4-pro

```
label                      P       R      F1      n
DATE                   0.990   0.971   0.981    209
GIVENNAME              0.691   0.663   0.677    199
SURNAME                0.681   0.725   0.703    171
CITY                   0.926   0.850   0.886    133
EMAIL                  1.000   1.000   1.000    131
TELEPHONENUM           0.858   0.990   0.920    104
STREET                 0.878   0.956   0.915     90
ZIPCODE                0.800   0.833   0.816     72
IDCARDNUM              0.422   0.811   0.555     53
CREDITCARDNUMBER       0.933   0.894   0.913     47
TAXNUM                 0.906   0.744   0.817     39
SOCIALNUM              0.821   0.657   0.730     35
--------------------------------------------------
MICRO                  0.818   0.849   0.833   1283

examples: 200  schema-invalid: 0 (0.0%)  hallucinated values: 1
```

**Total spend: $0.3875**
