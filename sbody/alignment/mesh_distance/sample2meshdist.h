#ifndef __s2m_h__
#define __s2m_h__

#include "Eigen/Core"

#include <tuple>
#include "robust.h"

using Eigen::Matrix3d;
typedef Eigen::RowVector3d V;

/*
  Point Plane Distance

  distance to the plane of triangle abc = n . (x-a) where n = (b-a) x (c-b) / |(b-a) x (c-b)| 
  expanding n gives 
     [(x-a) . (b-a) x (c-b)] / |(b-a) x (c-b)|
  the numerator is a scalar triple product
  and we can expand the denominator using |a x b| = |a||b|cos(t) = |a||b|sqrt(1-sin(t)^2)
  yielding
     det(x-a, b-a, c-b) / sqrt(|b-a|^2|c-b|^2 - ((b-a).(c-b))^2)
  which we will write
  det(A) / s
 
  d/dAij det(A) = (det(A)*inv(A))ij
 
  let x = b-a, y=c-b
  s = sqrt(|x|^2|y|^2 - (x.y)^2)
  ds/dx = 1/(2s) * [ 2x|y|^2 - 2(x.y)y ]
        = [ x|y|^2 - y(x.y) ] / s
  ds/dy = [ y|x|^2 - x(x.y) ] / s : by symmetry
  
  in terms of a,b,c
  ds/da = -ds/dx
  ds/db = ds/dx - ds/dy
  ds/dc = ds/dy
*/

// For some reason I get an undefined symbol error from Eigen when I try to use its cross product, so I've written my own
V cross(const V& a, const V& b) {
    return V(a(1)*b(2) - a(2)*b(1),
             b(0)*a(2) - b(2)*a(0),
             a(0)*b(1) - a(1)*b(0));
}

double det3(const Matrix3d& A) {
    return (A(0,0) * (A(1,1) * A(2,2) - A(2,1) * A(1,2))
           -A(1,0) * (A(0,1) * A(2,2) - A(2,1) * A(0,2))
           +A(2,0) * (A(0,1) * A(1,2) - A(1,1) * A(0,2)));
}
void det_times_inv3(const Matrix3d& A, Matrix3d& out) {
    out(0,0) = A(2,2)*A(1,1)-A(2,1)*A(1,2);
    out(0,1) = A(0,2)*A(2,1)-A(0,1)*A(2,2);
    out(0,2) = A(0,1)*A(1,2)-A(0,2)*A(1,1);
    
    out(1,0) = A(1,2)*A(2,0)-A(1,0)*A(2,2);
    out(1,1) = A(0,0)*A(2,2)-A(0,2)*A(2,0);
    out(1,2) = A(0,2)*A(1,0)-A(0,0)*A(1,2);

    out(2,0) = A(1,0)*A(2,1)-A(1,1)*A(2,0);
    out(2,1) = A(0,1)*A(2,0)-A(0,0)*A(2,1);
    out(2,2) = A(0,0)*A(1,1)-A(0,1)*A(1,0);
}

// computes f(dist) and derivatives
// f returns a tuple f(x),Df(x)
template<typename F>
double pointPlane(const double* x_, const double* a_, const double* b_, const double* c_,
        double* dx, double* da, double* db, double* dc, const F& f) {
    V x = V::Map(x_),
      a = V::Map(a_),
      b = V::Map(b_),
      c = V::Map(c_);

    Matrix3d A, ddetA;
    (A << x-a, b-a, c-b).finished();

    double detA=det3(A);
    det_times_inv3(A,ddetA);

    // here z stands in for the comment's x
    V z=b-a, y=c-b; 
    double s = cross(z,y).norm();
    V ds_a =  -(z*y.squaredNorm() - y*z.dot(y))/s,
      ds_c =   (y*z.squaredNorm() - z*z.dot(y))/s;
    V ds_b = -ds_a - ds_c;
    
    //double fv, Dfv;
    std::tuple<double, double> fv_Dfv = f(detA/s);
    //tie(fv,Dfv) = f(detA/s);

    double s_sq = s*s;
    if(dx) V::Map(dx) += std::get<1>(fv_Dfv) * (ddetA.col(0) / s).transpose(); 
    if(da) V::Map(da) += std::get<1>(fv_Dfv) * (((-ddetA.col(0)-ddetA.col(1)) / s).transpose() - (ds_a*(detA/s_sq)));
    if(db) V::Map(db) += std::get<1>(fv_Dfv) * (((ddetA.col(1)-ddetA.col(2)) / s).transpose() - (ds_b*(detA/s_sq)));
    if(dc) V::Map(dc) += std::get<1>(fv_Dfv) * ((ddetA.col(2) / s).transpose() - (ds_c*(detA/s_sq)));

    return std::get<0>(fv_Dfv);
}

/*
   Point-Line distance
     formula for distance function from http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
  
  dist(p, line supporting (a,b)) = |(p-a) x (p-b)|/|b-a|
  
  d/dpi dist = [(ei x (p-b) + (p-a) x ei)] . (p-a) x (p-b)]/ |(p-a) x (p-b)||b-a|]
   ei x [(p-b) + (a-p)] = ei x (a-b), so substitution yields
   yields ei x (a-b) . (p-a) x (p-b) / |(p-a) x (p-b)||b-a|
  d/dai: [(ei x (b-p) . [(p-a) x (p-b)]]/|(p-a) x (p-b)||b-a| +
               |(p-a) x (p-b)| * ei . (b-a) / |b-a|^3
  d/dbi: swap a and b in the expression for d/dai
         this amounts to replacing p-b with a-p in first line and negating b-a in the second
  
  r := (p-a) x (p-b) / |(p-a) x (p-b)||b-a| occurs in every expression and
  d := (b-a)|(p-a) x (p-b)|/|b-a|^3 occurs in most
  rewriting,
  d/dpi = ei x (a-b) . r
  d/dai = ei x (b-p) . r + ei . d
  d/dbi = ei x (p-a) . r - ei . d
  
  this can be simplified by swapping the . and the x usng a scalar-triple-product identity

  c . (a x b) = a . (b x c)  [ det(c,b,a) = det(a,b,c) ]
  
  d/dp = (a-b) x r
  d/da = (b-p) x r + d
  d/db = (p-a) x r - d

*/


template<typename F>
double pointLine(const double* x_, const double* a_, const double* b_,
        double* dx, double* da, double* db, const F& f) {
    V x = V::Map(x_),
      a = V::Map(a_),
      b = V::Map(b_);

    V w=cross(x-a,x-b); 
    double norm_w = w.norm(),
           norm_ab = (b-a).norm();
    double dist = norm_w / norm_ab;

    //double fv,Dfv;
    std::tuple<double, double> fv_Dfv(f(dist));
    //tie(fv,Dfv) = f(dist);
    
    V r = w/(norm_w*norm_ab),
      d = (b-a) * norm_w / (norm_ab*norm_ab*norm_ab);
   
    if(dx) V::Map(dx) += std::get<1>(fv_Dfv) * cross(a-b,r);
    if(da) V::Map(da) += std::get<1>(fv_Dfv) * (cross(b-x,r) + d);
    if(db) V::Map(db) += std::get<1>(fv_Dfv) * (cross(x-a,r) - d);
    return std::get<0>(fv_Dfv);
} 

template<typename F>
double pointPoint(const double* x_, const double* a_,
        double* dx, double* da, const F& f) {
    V x = V::Map(x_),
      a = V::Map(a_);
    double dist = (x-a).norm();

    //double fv, Dfv;
    //tie(fv,Dfv) = f(dist);
    std::tuple<double, double> fv_Dfv(f(dist));

    if(dx) V::Map(dx) += std::get<1>(fv_Dfv) * (x-a)/dist;
    if(da) V::Map(da) += std::get<1>(fv_Dfv) * (a-x)/dist;
    return std::get<0>(fv_Dfv);
}

// looping, wrappers

// for some reason i'm getting this: invalid conversion from ‘long long unsigned int*’ to ‘uint64_t*’
// when I pass the data from a numpy array of type n.uint64_t assuming it's a uint64_t
typedef long long unsigned int ui;
template<typename F>
struct Distance {
    F f;
    
    double plane(const double* x, const double* a, const double* b, const double* c,
                      double* dx, double* da, double* db, double* dc) {
        return pointPlane(x, a, b, c, dx, da, db, dc, f);
    }
    double line(const double* x, const double* a, const double* b, 
        double* dx, double* da, double* db) {
        return pointLine(x, a, b, dx, da, db, f);
    }
    double point(const double* x, const double* a, double* dx, double* da) {
        return pointPoint(x, a, dx, da, f);
    }

    double tri(int part, const double* x, const double* a, const double* b, const double* c,
        double* dx, double* da, double* db, double* dc) {
        switch(part) {
            case 0: return pointPlane(x, a, b, c, dx, da, db, dc, f);
            case 1: return pointLine(x, a, b, dx, da, db, f);
            case 2: return pointLine(x, b, c, dx, db, dc, f);
            case 3: return pointLine(x, c, a, dx, dc, da, f);
            case 4: return pointPoint(x, a, dx, da, f);
            case 5: return pointPoint(x, b, dx, db, f);
            case 6: return pointPoint(x, c, dx, dc, f);
        }
    }
};


namespace instances {

struct Distance : public ::Distance<Identity> {};
struct SquaredDistance : public ::Distance<Square> {};
struct GMDistance : public ::Distance< Compose<GM, Square> > {
    GMDistance(double sigma) {
        GM gm(sigma*sigma);
        f=Compose<GM,Square>(GM(sigma*sigma), Square());
    } 
};

}



#endif
