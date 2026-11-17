# Test-set results (2,500 held-out lines)

Lower is better. DER = share of letters with a wrong diacritic; WER = share of words with at least one wrong letter.
'no CE' = ignoring the last letter of each word (the grammatical case ending).

| System | DER | DER (no CE) | WER | WER (no CE) |
|---|---|---|---|---|
| letter_majority | 48.02% | 46.99% | 84.79% | 73.96% |
| word_lookup | 9.05% | 5.59% | 24.15% | 9.88% |
| tashkeel_linear (this model) | 4.71% | 3.19% | 14.81% | 7.33% |

Best epoch on validation: 15; weights 7.46 MB; run time 624 s.

## Error rate per diacritic (this model, test)

| Gold diacritic | Letters | Error rate |
|---|---|---|
| none | 75,939 | 0.77% |
| fatha | 150,213 | 3.53% |
| kasra | 64,448 | 5.84% |
| sukun | 62,656 | 1.58% |
| damma | 41,070 | 12.98% |
| shadda+fatha | 15,274 | 5.64% |
| kasratan | 4,349 | 16.62% |
| shadda+kasra | 3,119 | 14.91% |
| dammatan | 2,927 | 26.79% |
| fathatan | 3,200 | 9.81% |
| shadda+damma | 2,176 | 17.42% |
| shadda | 471 | 90.45% |
| shadda+kasratan | 273 | 27.11% |
| shadda+dammatan | 222 | 29.28% |
| shadda+fathatan | 132 | 15.91% |
