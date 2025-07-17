# Multithreaded Template Matcher with Scale-Invariant Normalized Cross-Correlation

**Raphael Bomshakian**

To use this script you need:
- A snippet of your target image in the working directory.
- A folder that contains the target image in the working directory.

An example set is provided courtesy of [Lorem Picsum](https://picsum.photos). I encourage you to use your own folder.

### Procedure
**1. Preprocess Cleaning**
<br>Produces a new folder of the cleaned search set by removing ICC profiles to ensure compatibility.

**2. Template Matching**
<br>Multithreaded matching phase. Simultaneously documents results in `match_results.csv`.

