#include <functional>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <stdexcept>

#ifndef __OPT_H__
#define __OPT_H__

//! Optimization namespace containing various optimization algorithms
/*!
	Currently implemented are the basic Barzilai-Borwein gradient descent methods 
	(<em><A HREF="https://en.wikipedia.org/wiki/Gradient_descent">see Wikipedia's 'Gradient Descent' article</A></em>) 
	for both scalar-valued functions of a single variable \f$f:R\longrightarrow R\f$ as well as scalar-valued functions of \f$d\f$ variables \f$ f:R^d\longrightarrow R\f$.
	<br/><br/>
	The gradient descent algorithms implement the following iterative method:
	\f[
		x_{n+1}=x_n-\gamma_n\nabla f(x_n),
	\f]
	where
	\f[
		\gamma_n=\frac{(x_n-x_{n-1})\cdot\left[\nabla f(x_n)-\nabla f(x_{n-1})\right]}{\left|\nabla f(x_n)-\nabla f(x_{n-1})\right|^2}
	\f]
	and
	\f$\nabla f\f$ is approximated with a simple finite-difference method using \f$\Delta x=\min(tol/10,\gamma_n/10)\f$, where \f$tol\f$ is a provided convergence tolerance.
	<br/><br/>
	The nonlinear conjugate gradient method implements the algorithm outlined in the <A HREF="https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method">
	Wikipedia article</A>, with the \f$\beta_n\f$ parameter coming from the Polak-Ribiere formula. Note that this implementation performs the line search step of
	the algorithm using lambda expressions, so that the library has an implicit dependency on the C++11 standard.
	<br/><br/>
	For functions of \f$d\f$ variables, a <b>std::vector<double></b> container is assumed as the vector argument.
	<br/><br/>
	The following presents a simple example of using these optimizers.
	<br/>
	\include maintest.cpp
	<br/><br/>
	This program, ran using VS compiler on a Windows 10 64-bit machine, output the following:
	<br/>
	\code{.unparsed}
	Minimum converged in 2 iterations
	Solution for min val of f(x) = x^2 with initial guess x0 = 8:
			Minimum = 2.5e-15, x_min = -5e-08

	Minimum converged in 2 iterations
	Solution for min val of f(x) = |x|^2 with initial guess X0 = [4, 4]:
			Minimum = 5e-15, x_min = [-5e-08, -5e-08]

	Minimum converged in 55 iterations
	Solution for min val of rosenbrock(x) with initial guess X0 = [-0.5, 0.5]:
        Minimum = 9.83781e-11, x_min = [1, 1]

	Minimum converged in 20 iterations
	Solution for min val of rosenbrock(x) with initial guess X0 = [-0.5, 0.5]:
		Minimum = 1.97327e-12, x_min = [1, 1]
	\endcode
*/
namespace opt {

	namespace __helpers {
		std::vector<double> vectSum(std::vector<double> x1, const std::vector<double> x2)
		{
			if (x1.size() != x2.size())
				std::runtime_error::runtime_error("std::vector<double> operator+ only defined for vectors of equal size!\n");
			std::vector<double> ans = x1;
			size_t N = x1.size();
			for (size_t i = 0; i < N; i++)
				ans[i] += x2[i];
			return ans;
		}

		std::vector<double> vectMinus(std::vector<double> x1, const std::vector<double> x2)
		{
			if (x1.size() != x2.size())
				std::runtime_error::runtime_error("std::vector<double> operator- only defined for vectors of equal size!\n");
			std::vector<double> ans = x1;
			size_t N = x1.size();
			for (size_t i = 0; i < N; i++)
				ans[i] -= x2[i];
			return ans;
		}

		std::vector<double> vectScale(const double c, std::vector<double> vec)
		{
			size_t N = vec.size();
			std::vector<double> ans = vec;
			for (size_t i = 0; i < N; i++)
				ans[i] *= c;
			return ans;
		}

		double vectDot(std::vector<double> x1, const std::vector<double> x2)
		{
			if (x1.size() != x2.size())
				std::runtime_error::runtime_error("std::vector<double> dot product only defined for vectors of equal size!\n");
			size_t N = x1.size();
			double ans = 0.;
			for (size_t i = 0; i < N; i++)
				ans += x1[i]*x2[i];
			return ans;
		}

		std::vector<double> grad_Xd(const std::function<double(std::vector<double>)>f, const std::vector<double> x, const double h)
		{
			size_t N = x.size();
			std::vector<double> ans(N,0.), x_plus_h;
			for (size_t i = 0; i < N; i++)
			{
				x_plus_h = x;
				x_plus_h[i] += h;
				ans[i] = f(x_plus_h) - f(x);
				ans[i] /= h;
			}
			return ans;
		}

		double grad(const std::function<double(double)>f, const double x, const double h)
		{
			return ( f(x + h) - f(x) ) / h;
		}
	}

	//! Compute a local minimum near initial guess of real-valued function
	/*!
	This function implements the unconstrained Barzilai-Borwein gradient descent method (<em><A HREF="https://en.wikipedia.org/wiki/Gradient_descent">see Wikipedia's 'Gradient Descent' article</A></em>) for a general single variable real-valued function.
	\param f <b>(std::function<double(double)>)</b> real-valued function \f$f:R\longrightarrow R\f$ to find the local minimum nearest initial guess \f$x_0\f$. \f$f\f$ is a function that has <b>double</b> arg type, and <b>double</b> as return type.
	\param x0 <b>(double&)</b> Initial guess of gradient descent optimization problem; this parameter is updated with the optimized \f$x-\f$value prior to exiting.
	\param tol <b>(double)</b> by default 1.0e-6, the convergence criterion for function derivative magnitude
	\param verbose <b>(bool)</b> by default true, if set to true then the optimizer will echo to stdout number of iterations performed
	\return \f$f(x_{opt})\f$ <b>(double)</b> local minimum of \f$f\f$ in well of \f$x_0\f$
	*/
	double gradientDescent(std::function<double(double)>f, double& x0, double tol = 1.e-6, bool verbose = true)
	{
		double error = tol + 1., x_np1 = x0, x_n = x0, x_np1_temp = x0, x_nm1 = x0;
		double dx = tol / 10., df_nm1 = 1., df_n = 1., gamma_n = 1.0;
		size_t iterations = 0;

		df_n = __helpers::grad(f, x_n, dx);
		x_np1 = x_n - gamma_n * df_n;
		x_nm1 = x_n;
		x_n = x_np1;

		while (error > tol)
		{
			df_nm1 = df_n;								// Update gradient of f at previous step
			df_n = __helpers::grad(f, x_n, dx);			// Update gradient of f at current step
			gamma_n = (x_n - x_nm1) * (df_n - df_nm1)	// Update gamma at current step
				/ ((df_n - df_nm1) * (df_n - df_nm1));
			x_np1 = x_n - gamma_n * df_n;				// Update x_{n+1} (main algorithm step)
			x_nm1 = x_n;								// Update previous x_{n-1} with x_n
			x_n = x_np1;								// Update x_n with new x_{n+1}
			dx = fmin(tol / 10., gamma_n/10.);			// Update dx based on new spatial step value

			error = fabs(df_n);							// Update error based on new function evaluations
			iterations++;								// Increment algorithm iteration count
		}
		x0 = x_n;
		if (verbose) {
			std::cout << "Minimum converged in " << iterations << " iterations" << std::endl;
			std::cout << "\tMinimum = " << f(x_n) << ", x_min = " << x0 << std::endl;
		}
		return f(x0);
	};

	//! Compute a local minimum near initial guess of real-valued function
	/*!
	This function implements the unconstrained Barzilai-Borwein gradient descent method (<em><A HREF="https://en.wikipedia.org/wiki/Gradient_descent">see Wikipedia's 'Gradient Descent' article</A></em>) for a general multi-variate real-valued function.
	\param f <b>(std::function<double(std::vector<double>)>)</b> real-valued function \f$ f:R^d\longrightarrow R\f$ to find the local minimum nearest initial guess \f$x_0\f$. \f$f\f$ is a function that has <b>std::vector<double></b> arg type, and <b>double</b> as return type.
	\param x0 <b>(double&)</b> Initial guess of gradient descent optimization problem; this parameter is updated with the optimized \f$x-\f$value prior to exiting.
	\param tol <b>(double)</b> by default 1.0e-6, the convergence criterion for function gradient magnitude
	\param verbose <b>(bool)</b> by default true, if set to true then the optimizer will echo to stdout number of iterations performed
	\return \f$f(x_{opt})\f$ <b>(double)</b> local minimum of \f$f\f$ in well of \f$x_0\f$
	*/
	double gradientDescent_Xd(std::function<double(std::vector<double>)>f, std::vector<double>& x0, double tol = 1.e-6, bool verbose = true)
	{
		std::vector<double>  x_np1 = x0, x_n = x0, x_np1_temp = x0, x_nm1 = x0, df_nm1, df_n;
		double error = tol + 1., dx = tol / 10., gamma_n = 1.0;
		size_t iterations = 0;

		df_n = __helpers::grad_Xd(f, x_n, dx);
		x_np1 = __helpers::vectMinus(x_n,__helpers::vectScale(gamma_n, df_n));
		x_nm1 = x_n;
		x_n = x_np1;

		while (error > tol)
		{
			df_nm1 = df_n;									// Update gradient of f at previous step
			df_n = __helpers::grad_Xd(f, x_n, dx);			// Update gradient of f at current step
			gamma_n = 
				__helpers::vectDot(
					__helpers::vectMinus(x_n,x_nm1),
					__helpers::vectMinus(df_n, df_nm1)) / 
			  ( __helpers::vectDot(
					__helpers::vectMinus(df_n,df_nm1),
				    __helpers::vectMinus(df_n, df_nm1)));	// Update gamma at current step
			x_np1 = __helpers::vectMinus(x_n,
				__helpers::vectScale(gamma_n, df_n));		// Update x_{n+1} (main algorithm step)
			x_nm1 = x_n;									// Update previous x_{n-1} with x_n
			x_n = x_np1;									// Update x_n with new x_{n+1}
			dx = fmin(tol / 10., gamma_n/10.);				// Update dx based on new spatial step value

			error = fabs(__helpers::vectDot(df_n, df_n));	// Update error based on new function evaluations
			iterations++;									// Increment algorithm iteration count
		}
		x0 = x_n;
		if (verbose) {
			std::cout << "Minimum converged in " << iterations << " iterations" << std::endl;
			std::cout << "\tMinimum = " << f(x_n) << ", x_min = [";
			for (size_t i = 0; i < x0.size() - 1; i++)
				std::cout << x0[i] << ", ";
			std::cout << x0[x0.size() - 1] << "]" << std::endl;
		}
		return f(x0);
	};

	//! Nonlinear conjugate gradient descent algorithm of real-valued function
	/*!
	This function implements an unconstrained nonlinear conjugate gradient descent method (<em><A HREF="https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method">see Wikipedia's article</A></em>) for a general multi-variate real-valued function. The line search performed for updating the solution guess (see the \f$\alpha_n\f$ parameter in the Wikipedia article) uses the opt::gradientDescent() method. \f$\beta_n\f$ is computed using the Polak-Ribiere formula, also found in the Wikipedia article.
	\param f <b>(std::function<double(std::vector<double>)>)</b> real-valued function \f$ f:R^d\longrightarrow R\f$ to find the local minimum nearest initial guess \f$x_0\f$. \f$f\f$ is a function that has <b>std::vector<double></b> arg type, and <b>double</b> as return type.
	\param x0 <b>(double&)</b> Initial guess of optimization problem; this parameter is updated with the optimized \f$x-\f$value prior to exiting.
	\param tol <b>(double)</b> by default 1.0e-6, the convergence criterion for function gradient magnitude
	\param verbose <b>(bool)</b> by default true, if set to true then the optimizer will echo to stdout number of iterations performed
	\return \f$f(x_{opt})\f$ <b>(double)</b> local minimum of \f$f\f$ in well of \f$x_0\f$
	\todo Allow for user options to be set, such as which formula to use for the \f$\beta_n\f$ calculations
	\todo Solutions for example problems seem to go 'nan' if tolerance is set any lower than 1.e-6; need to investigate
	*/
	double ncgd_Xd(std::function<double(std::vector<double>)>f, std::vector<double>& x0, double tol = 1.e-6, bool verbose = true)
	{
		std::vector<double>  
			Deltax_n,			// Current step direction
			Deltax_nm1,			// Previous step direction
			s_n,				// Current conjugate direction
			s_nm1,				// Previous conjugate direction
			x_n = x0,			// Current solution guess
			x_np1 = x0;			// Previous solution guess
		double 
			error = tol + 1.,	// Current error metric
			dx = tol / 10.,		// Spatial step size for computing gradient
			alpha_n = 0.0,		// Optimized line step
			beta_n = 0.0,		// Conjugate direction update scalar
			linesearch;			// Placeholder variable for result of line search
		size_t iterations = 0;	// Number of iterations performed
		
		// Update Deltax_n with (negative) gradient of f
		Deltax_n = __helpers::vectScale(-1.0,					
			__helpers::grad_Xd(f, x0, dx));

		// Use beta_n = 0 for first iteration
		s_n = Deltax_n;

		// perform line search using gradientDescent, updates alpha_n by reference
		linesearch = gradientDescent(							
			[f, x_n, s_n](double alpha) mutable { return f(__helpers::vectSum(x_n, __helpers::vectScale(alpha, s_n))); },
			alpha_n,tol,false);

		// Update next x_{n+1}
		x_np1 = __helpers::vectSum(x_n,							
			__helpers::vectScale(alpha_n,s_n));

		// Update previous step values
		Deltax_nm1 = Deltax_n;									
		s_nm1 = s_n;

		// Update error based on new function evaluations
		error = fabs(__helpers::vectDot(Deltax_n, Deltax_n));	

		// Increment algorithm iteration count
		iterations++;											

		while (error > tol)
		{
			// Update Deltax_n with (negative) gradient of f
			Deltax_n = __helpers::vectScale(-1.0,					
				__helpers::grad_Xd(f, x_n, dx));

			// Update beta_n via Polak-Ribiere formula
			beta_n =												
				__helpers::vectDot(Deltax_n,
					__helpers::vectMinus(Deltax_n, Deltax_nm1)) /
				__helpers::vectDot(Deltax_nm1, Deltax_nm1);

			// Update conjugate direction
			s_n = __helpers::vectSum(Deltax_n,						
				__helpers::vectScale(beta_n,s_nm1));

			// reset alpha_n from previous iteration
			alpha_n = 0.;											

			// perform line search using gradientDescent, updates alpha_n by reference
			linesearch = gradientDescent(							
				[f, x_n, s_n](double alpha) mutable { return f(__helpers::vectSum(x_n, __helpers::vectScale(alpha, s_n))); },
				alpha_n, tol, false);

			// Update next x_{n+1}
			x_np1 = __helpers::vectSum(x_n,							
				__helpers::vectScale(alpha_n, s_n));

			// Update previous step values
			Deltax_nm1 = Deltax_n;									
			s_nm1 = s_n;
			x_n = x_np1;

			// Update dx based on new spatial step value
			dx = fmin(tol / 10., alpha_n / 10.);					

			// Update error based on new function evaluations
			error = fabs(__helpers::vectDot(Deltax_n, Deltax_n));	

			// Increment algorithm iteration count
			iterations++;											
		}
		x0 = x_n;
		if (verbose) {
			std::cout << "Minimum converged in " << iterations << " iterations" << std::endl;
			std::cout << "\tMinimum = " << f(x_n) << ", x_min = [";
			for(size_t i = 0; i < x0.size()-1;i++)
				std::cout << x0[i] <<", ";
			std::cout << x0[x0.size() - 1] << "]" << std::endl;
		}
		return f(x0);
	};
};
#endif