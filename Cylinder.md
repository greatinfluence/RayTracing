$$
||\text{Perp}_{u}(p+td-c)||^2=r^2
$$

$$
||(p+td-c)-u(u^Tu)^{-1}u^T(p+td-c)||^2=r^2
$$

Apply the pre-restriction that $||u||=||d||=1$ :
$$
||(p+td-c)-uu^T(p+td-c)||^2=r^2
$$
Denote $s=p-c$ :
$$
[(s+td)-uu^T(s+td)]^T[(s+td)-uu^T(s+td)]=r^2\\
||s+td||^2-\lang s+td, u\rang^2-\lang s+td, u\rang^2+(s+td)^Tu(u^Tu)u^T(s+td)\\
=||s+td||^2-\lang s+td, u\rang^2=r^2
$$

$$
\lang s+td,s+td\rang-\lang s+td, u\rang^2=r^2\\
t^2||d||^2+2t\lang s, d\rang+||s||^2-(\lang s, u\rang+t\lang d, u\rang)^2=r^2\\
t^2||d||^2+2t\lang s, d\rang+||s||^2-\lang d, u\rang^2t^2-2t\lang s, u\rang \lang d, u\rang-\lang s, u\rang^2-r^2=0
$$

$$
(1-\lang d,u\rang^2)t^2+2(\lang s, d\rang - \lang s, u\rang \lang d, u\rang)t+||s||^2-\lang s, u\rang ^2-r^2=0
$$

