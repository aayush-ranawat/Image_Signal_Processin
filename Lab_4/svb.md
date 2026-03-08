# Calculation of Parameters A and B for Space-Variant Blurring

We are given the standard deviation function for space-variant blurring:

$$\sigma(m, n) = A \exp\left( - \frac{(m - \frac{N}{2})^2 + (n - \frac{N}{2})^2}{B} \right)$$

Given the size of the image is $195 \times 195$, we set $N = 195$. The center of the image is $\frac{195}{2} = 97.5$. 

We must find $A$ and $B$ using the two provided constraints.

## 1. Solving for A

**Constraint 1:** The standard deviation at the center of the image is $2.0$.

$$\sigma(97.5, 97.5) = 2.0$$

Substitute $m = 97.5$ and $n = 97.5$ into the original equation:

$$2.0 = A \exp\left( - \frac{(97.5 - 97.5)^2 + (97.5 - 97.5)^2}{B} \right)$$

Simplify the numerator inside the exponential:

$$2.0 = A \exp\left( - \frac{0 + 0}{B} \right)$$

$$2.0 = A \exp(0)$$

Since $\exp(0) = 1$, we find $A$:

$$A = 2.0$$

---

## 2. Solving for B

**Constraint 2:** The standard deviation at the top-left corner of the image is $0.01$.

$$\sigma(0, 0) = 0.01$$

Substitute $m = 0$, $n = 0$, and our known value $A = 2.0$ into the equation:

$$0.01 = 2.0 \exp\left( - \frac{(0 - 97.5)^2 + (0 - 97.5)^2}{B} \right)$$

Divide both sides by $2.0$:

$$0.005 = \exp\left( - \frac{(-97.5)^2 + (-97.5)^2}{B} \right)$$

Square the terms in the numerator ($-97.5^2 = 9506.25$):

$$0.005 = \exp\left( - \frac{9506.25 + 9506.25}{B} \right)$$

Combine the fractions:

$$0.005 = \exp\left( - \frac{19012.5}{B} \right)$$

To isolate $B$, take the natural logarithm ($\ln$) of both sides:

$$\ln(0.005) = - \frac{19012.5}{B}$$

Multiply both sides by $B$ and divide by $\ln(0.005)$:

$$B = - \frac{19012.5}{\ln(0.005)}$$

Calculating the numerical value ($\ln(0.005) \approx -5.2983$):

$$B \approx \frac{-19012.5}{-5.2983} \approx 3588.40$$

---

## Final Parameters

* $A = 2.0$
* $B = - \frac{19012.5}{\ln(0.005)} \approx 3588.40$