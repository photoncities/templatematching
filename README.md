# Multithreaded Template Matcher with Scale-Invariant Normalized Cross-Correlation

**Raphael Bomshakian**

To use this script you need:
- A snippet of your target image in the working directory.
- A folder that contains the target image in the working directory.

The following packages are necessary:
```
opencv-python numpy PIL 
```

An example set is provided courtesy of [Lorem Picsum](https://picsum.photos). I encourage you to use your own folder.

### Procedure
**1. Preprocess Cleaning**
<br>Produces a new folder of the cleaned search set by removing ICC profiles to ensure compatibility.

**2. Template Matching**
<br>Multithreaded matching phase. Simultaneously documents results in `match_results.csv`.

I'm aware the documentation process is a bit unique; Additions to the match_results are thread-safely sorted in ascending order, and when complete, sorted in descending order. I did this so someone viewing the csv during the process can see the current top match (turn autoscrolling on if you have it in your working terminal) and exit early if necessary.
