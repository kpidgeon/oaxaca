# oaxaca

This module will implement Oaxaca-Blinder decomposition. and is inspired by the package 'oaxaca' in R (see https://cran.r-project.org/web/packages/oaxaca/vignettes/oaxaca.pdf).
I aim to provide much of the same functionality, but in Python. There exists a module called 'oaxaca' within statsmodels, but it does not provide some of the functionality that
I desire. So I have decided to write my own implementation as a bit of an exercise, not really in statistics or in linear algebra as the maths (at least for a basic understanding
of the process) is straightforward; it's more an exercise in designing a useful, reasonably clean piece of code.

The intended features will include:
- Two-fold Oaxaca-Blinder decomposition
- Three-fold Oaxaca-Blinder decomposition
- Several reference coefficient weightings for two-fold decomposition (see the R implementation above)
- Insensitivity to omitted baseline dummies (see R paper again)
- Bootstrapping methods for estimating accuracy from an empirical distribution
- Plotting (by decomposition components, variables, or aggregate values)

