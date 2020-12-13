#ifndef __robust_h__
#define __robust_h__

#include <tuple>

/*
 * Robust estimators
 */

// the geman-mcclure robust m-estimator
// because we'll be operating on both distances and squared distances
// versions will be defined for squared and non-squared input
// takes squared input: x^2 -> s^2 x^2 / (s^2 + x^2)
struct GM {
    double sigmasq;
    GM() : sigmasq(1) {}
    GM(double sigmasq_) : sigmasq(sigmasq_) {}
    // return rho(x) and Drho/d[x^2] (x)
    std::tuple<double,double> operator()(double xsq) const {
        return std::make_tuple(sigmasq * xsq / (sigmasq + xsq),
                          sigmasq * (1 / (sigmasq + xsq)) - sigmasq*xsq / ((sigmasq + xsq)*(sigmasq + xsq)));
    }
};

struct Identity {
    std::tuple<double,double> operator()(double x) const {
        return std::make_tuple(x, 1);
    }
};

struct Square {
    std::tuple<double,double> operator()(double x) const {
        return std::make_tuple(x*x, 2*x);
    }
};

template<typename F,typename G>
struct Compose {
    F f;
    G g;
    Compose() {}
    Compose(const F& f_, const G& g_) : f(f_), g(g_) {}
    std::tuple<double,double> operator()(double x) const {
        //double gx,Dgx,fx,Dfx;
        std::tuple<double, double> gx_Dgx = g(x);
        std::tuple<double, double> fx_Dfx = f(std::get<0>(gx_Dgx));
        return std::make_tuple(std::get<0>(fx_Dfx),std::get<1>(fx_Dfx)*std::get<1>(gx_Dgx));
    }
};

#endif
