/*! 
	\author Nicholas Crabb

\mainpage basic_opt: a simple C++ optimization library
 This library is intended to provide a simple, user-friendly C++ black-box optimization library.
 <br/>
 The current user-interface is through a single header file opt.h.  
 <br/>
 The optimization functions are within the <A HREF="https://charelstoncrabb.github.io/basic_opt/html/namespaceopt.html">opt</A> namespace.
 <br/>
 Currently implemented is a simple <A HREF="https://en.wikipedia.org/wiki/Gradient_descent">gradient descent method</A> for single- or multi-variate functions, as well as the <A HREF="https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method">non-linear conjugate gradient version</A>.
 <br/>
 Up next, writing <A HREF="http://eigen.tuxfamily.org/index.php?title=Main_Page">Eigen</A>-friendly implementations of these optimization methods.
*/