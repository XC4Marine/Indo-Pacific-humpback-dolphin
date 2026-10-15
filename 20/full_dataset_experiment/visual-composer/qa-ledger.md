# Render QA ledger

| Issue | Artifact | Page/section | Severity | Fix | Owner | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Unicode subscripts missing in selected font | Individual/overview figures | Atom norm and amplitude labels | Low | Replaced Unicode subscripts with portable `L2` and `a0` notation | Visual composer | Fixed |
| Case metadata overlapped the first panel title | Individual figures | Top margin | Medium | Reserved a dedicated title/subtitle band above the plotting grid | Visual composer | Fixed |
| Arrival markers might drift from stored delays | All case figures | Raw and filtered waveforms | High | Recomputed each delay from `(arrival_sample - direct_sample) / 576000`; maximum discrepancy below `7e-17` ms | Visual composer | Passed |
| Extracted atom normalization | Four selected atoms | Atom waveform panels | High | Verified each saved atom has L2 norm equal to 1 within floating-point precision | Visual composer | Passed |
| Editable vector text | Five SVG files | Whole figure | Medium | Verified live `<text>` elements in every SVG | Visual composer | Passed |
