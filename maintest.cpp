#include <vector>
#include <iostream>
#include <stdexcept>
#include "opt.h"

// Single-variable f(x) = x^2
double x2(double x)
{
	return x * x;
}

// Multi-variate f(x) = |x|^2
double x2_Xd(std::vector<double> x)
{
	size_t N = x.size();
	double ans = 0.0;
	for (size_t i = 0; i < N; i++)
		ans += x[i] * x[i];
	return ans;
}

// Rosenbrock function
double rosenbrock(std::vector<double> x)
{
	if (x.size() != 2)
		std::runtime_error::runtime_error("rosenbrock function only takes 2D vectors as input");

	return (1 - x[0])*(1 - x[0]) + 100.0*(x[1] - x[0] * x[0])*(x[1] - x[0] * x[0]);
}

int main()
{
	// Single-variable problem parameters:
	double x0 = 8.;
	double x0_init = x0;

	/*** Call the single-variable gradient descent optimizer on the x^2 function: ***/
	double gdval = opt::gradientDescent(x2, x0);

	// Output results:
	std::cout << "Solution for min val of f(x) = x^2 with initial guess x0 = " << x0_init << ":" << std::endl;
	std::cout <<  "\tMinimum = " << gdval << ", x_min = " << x0 << std::endl;

	std::cout << std::endl;

	// Multi-variable problem parameters:
	std::vector<double> X0 = { 4.,4. };
	std::vector<double> X0_init = X0;

	/*** Call the multi-variable gradient descent optimizer on the |x|^2 function: ***/
	gdval = opt::gradientDescent_Xd(x2_Xd, X0);

	// Output results:
	std::cout << "Solution for min val of f(x) = |x|^2 with initial guess X0 = [" << X0_init[0] << ", " << X0_init[1] << "]:" << std::endl;
	std::cout << "\tMinimum = " << gdval << ", x_min = [" << X0[0] << ", " << X0[1] << "]" << std::endl;

	std::cout << std::endl;

	// Re-initialize parameters for rosenbrock opt problem
	X0 = {-0.5, 0.5};
	X0_init = X0;

	/*** Call the multi-variable gradient descent optimizer on the Rosenbrock function: ***/
	gdval = opt::gradientDescent_Xd(rosenbrock, X0);

	// Output results:
	std::cout << "Solution for min val of rosenbrock(x) with initial guess X0 = [" << X0_init[0] << ", " << X0_init[1] << "]:" << std::endl;
	std::cout << "\tMinimum = " << gdval << ", x_min = [" << X0[0] << ", " << X0[1] << "]" << std::endl;
	std::cin.get();
	return 0;
}