# Multithreaded Template Matcher with Scale-Invariant Normalized Cross-Correlation

To use this script you need:
- A snippet of your target image
- A folder that contains the target image.

### Procedure
**1. Cleaning**
<br>Produces a new folder of the cleaned search set by removing ICC profiles to ensure compatibility.

**2. Template Matching**
<br>Multithreaded matching phase. Simultaneously documents results in `match_results.csv`.

