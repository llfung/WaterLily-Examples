## Formulas for inertial torque of a spheroid.
#  Derived from first order Re expansion from Stokes flow
#  Formulas from Dabade et al. (2016, JFM) The effect of inertia on the orientation dynamics of anisotropic particles in simple shear flow, 10.1017/jfm.2016.14

# Aspect ratio of spheroid
AR=3.0;

# Eccentricity of spheroid
e=sqrt(1. -1. / AR^2.);

# Shape factor for torque
F_fac=pi*e^2. /(315. *((e^2. +1. )*atanh(e)-e)^2. * ((1. -3. * e^2. )*atanh(e)-e));
F=(-(420. *e + 2240. *e^3. +4249. *e^5. -2152. *e^7. ) +(420. +3360. *e^2. +1890. *e^4. -1470. *e^6. )*atanh(e) -(1260. *e -1995. *e^3. +2730. *e^5. -1995. *e^7. )*atanh(e)^2)*F_fac;

# Angle between uniform flow and spheroid axis (in radians)
angle = pi/4;

# Final torque formula (in units of ρU²L³)
F*sin(angle*2)/2