/*
Purpose
-------
Standalone tester for the exact Hermitian dot product and Hermitian cross product
implemented over Q(√n) with complex coefficients. This mirrors the logic used in
the solver so you can validate inputs and behavior independently.

Build
-----
  g++ -std=c++17 -O2 -o test_complex_ops cadical-rcl/src/test_complex_ops.cpp

Usage
-----
  ./test_complex_ops a1 a2 a3 b1 b2 b3

Each argument is a component (string) of a complex 3-vector. Components can use:
  - Integers:            1, -3
  - Rationals:           1/2, -3/4
  - Square roots:        sqrt(2), 3/4*sqrt(5), -1/2*sqrt(3)
  - Combinations:        1 + 1/2*sqrt(2), -1/3 - 2*sqrt(7)
  - Imaginary unit I:    I, -I, 1 + I, 1/2*sqrt(3) - I

Spacing is ignored, e.g. "-1/2*sqrt(2) - I" is accepted.

Examples
--------
  # Simple integer/imag examples
  ./test_complex_ops "1" "-I" "-I-1" "1" "I" "-I+1"

  # With rationals and sqrt
  ./test_complex_ops "-1/2*sqrt(2)" "0" "1/3" "I" "-1/2*sqrt(3)" "1/2+sqrt(2)"

Output
------
Prints the exact Hermitian dot result, the three components of the Hermitian cross,
and a final yes/no indicating orthogonality (dot == 0 as a complex quadratic number).
*/

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <numeric>

// Standalone tester for exact Hermitian dot and Hermitian cross over Q(√n)

struct Rational { long long num; long long den; };
struct Quadratic { Rational a; Rational b; int rad; };         // a + b√rad
struct ComplexQuadratic { Quadratic re; Quadratic im; };       // re + i im

// Utilities
static long long gcd(long long a, long long b) {
    while (b != 0) {
        long long temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

static Rational rational_normalize(Rational r) {
    if (r.den < 0) { r.den = -r.den; r.num = -r.num; }
    if (r.num == 0) { r.den = 1; return r; }
    long long g = gcd(std::llabs(r.num), std::llabs(r.den));
    if (g > 1) { r.num /= g; r.den /= g; }
    return r;
}
static Rational R(long long n, long long d = 1) { return rational_normalize({n, d}); }
static bool rational_is_zero(const Rational &x) { return x.num == 0; }
static Rational rational_add(const Rational &x, const Rational &y) { return rational_normalize({ x.num*y.den + y.num*x.den, x.den*y.den }); }
static Rational rational_sub(const Rational &x, const Rational &y) { return rational_normalize({ x.num*y.den - y.num*x.den, x.den*y.den }); }
static Rational rational_mul(const Rational &x, const Rational &y) { return rational_normalize({ x.num*y.num, x.den*y.den }); }
static Rational rational_neg(const Rational &x) { return { -x.num, x.den }; }

static bool quadratic_is_zero(const Quadratic &q) { return rational_is_zero(q.a) && rational_is_zero(q.b); }
static Quadratic quadratic_add(const Quadratic &x, const Quadratic &y) { return Quadratic{ rational_add(x.a,y.a), rational_add(x.b,y.b), x.rad }; }
static Quadratic quadratic_sub(const Quadratic &x, const Quadratic &y) { return Quadratic{ rational_sub(x.a,y.a), rational_sub(x.b,y.b), x.rad }; }
static Quadratic quadratic_mul(const Quadratic &x, const Quadratic &y) {
    // (a1 + b1√r)(a2 + b2√r) = (a1a2 + b1b2 r) + (a1b2 + a2b1)√r
    Rational a1a2 = rational_mul(x.a, y.a);
    Rational b1b2 = rational_mul(x.b, y.b);
    Rational rterm{ b1b2.num * y.rad, b1b2.den };
    Rational a = rational_add(a1a2, rterm);
    Rational b = rational_add(rational_mul(x.a, y.b), rational_mul(y.a, x.b));
    return Quadratic{ rational_normalize(a), rational_normalize(b), x.rad };
}

static ComplexQuadratic cq_conj(const ComplexQuadratic &z) { return ComplexQuadratic{ z.re, Quadratic{ rational_neg(z.im.a), rational_neg(z.im.b), z.im.rad } }; }
static ComplexQuadratic cq_add(const ComplexQuadratic &x, const ComplexQuadratic &y) { return ComplexQuadratic{ quadratic_add(x.re,y.re), quadratic_add(x.im,y.im) }; }
static ComplexQuadratic cq_sub(const ComplexQuadratic &x, const ComplexQuadratic &y) { return ComplexQuadratic{ quadratic_sub(x.re,y.re), quadratic_sub(x.im,y.im) }; }
static ComplexQuadratic cq_mul(const ComplexQuadratic &x, const ComplexQuadratic &y) {
    // (xr + i xi)(yr + i yi) = (xr*yr - xi*yi) + i(xr*yi + xi*yr)
    Quadratic real = quadratic_sub(quadratic_mul(x.re,y.re), quadratic_mul(x.im,y.im));
    Quadratic imag = quadratic_add(quadratic_mul(x.re,y.im), quadratic_mul(x.im,y.re));
    return ComplexQuadratic{ real, imag };
}
static bool cq_is_zero(const ComplexQuadratic &z) { return quadratic_is_zero(z.re) && quadratic_is_zero(z.im); }

// Hermitian dot and cross (length-3 vectors)
static ComplexQuadratic dot_complex_cq(const std::array<ComplexQuadratic,3>& a, const std::array<ComplexQuadratic,3>& b) {
    ComplexQuadratic sum{ Quadratic{ R(0), R(0), 2 }, Quadratic{ R(0), R(0), 2 } };
    for (int i = 0; i < 3; i++) sum = cq_add(sum, cq_mul(cq_conj(a[i]), b[i]));
    return sum;
}
static std::array<ComplexQuadratic,3> cross_complex_cq(const std::array<ComplexQuadratic,3>& a, const std::array<ComplexQuadratic,3>& b) {
    ComplexQuadratic x = cq_sub(cq_mul(cq_conj(a[1]), b[2]), cq_mul(cq_conj(a[2]), b[1]));
    ComplexQuadratic y = cq_sub(cq_mul(cq_conj(a[2]), b[0]), cq_mul(cq_conj(a[0]), b[2]));
    ComplexQuadratic z = cq_sub(cq_mul(cq_conj(a[0]), b[1]), cq_mul(cq_conj(a[1]), b[0]));
    return {x,y,z};
}

// Parsing: rationals and a ± b*sqrt(n), and combinations with I in a single token
static Rational rational_from_string(const std::string& s) {
    std::string t; for (char c : s) if (c != ' ') t.push_back(c);
    size_t slash = t.find('/');
    if (slash == std::string::npos) return rational_normalize({ std::stoll(t), 1 });
    return rational_normalize({ std::stoll(t.substr(0,slash)), std::stoll(t.substr(slash+1)) });
}
static Quadratic parse_quadratic_real(const std::string& token) {
    std::string t; for (char c : token) if (c != ' ') t.push_back(c);
    size_t sq = t.find("sqrt(");
    if (sq == std::string::npos) return Quadratic{ rational_from_string(t), R(0), 2 };
    size_t split = std::string::npos; for (size_t i = 1; i < t.size(); i++) if (t[i]=='+'||t[i]=='-') split = i;
    Rational a = R(0), b = R(0); int rad = 2;
    if (split == std::string::npos) {
        std::string coef = t.substr(0, sq);
        if (coef.empty() || coef == "+") b = R(1); else if (coef == "-") b = R(-1); else b = rational_from_string(coef);
    } else {
        std::string left = t.substr(0, split), right = t.substr(split);
        if (left.find("sqrt(") != std::string::npos) {
            std::string coef = left.substr(0, sq);
            if (coef.empty() || coef == "+") b = R(1); else if (coef == "-") b = R(-1); else b = rational_from_string(coef);
            a = rational_from_string(right);
        } else {
            a = rational_from_string(left);
            std::string s2 = right; size_t sq2 = s2.find("sqrt(");
            std::string coef = s2.substr(0, sq2);
            if (coef == "+" || coef.empty()) b = R(1); else if (coef == "-") b = R(-1); else b = rational_from_string(coef);
        }
    }
    size_t open = t.find('('), close = t.find(')');
    if (open != std::string::npos && close != std::string::npos && close > open+1) rad = std::stoi(t.substr(open+1, close-open-1));
    return Quadratic{ rational_normalize(a), rational_normalize(b), rad };
}
static ComplexQuadratic parse_cq_component(const std::string& token) {
    std::string t; for (char c : token) if (c != ' ') t.push_back(c);
    if (t.find('I') == std::string::npos) return ComplexQuadratic{ parse_quadratic_real(t), Quadratic{ R(0), R(0), 2 } };
    std::vector<std::string> terms; size_t i=0,n=t.size();
    while (i<n) { size_t j=i; if (t[j]=='+'||t[j]=='-') j++; while (j<n && t[j]!='+' && t[j]!='-') j++; terms.push_back(t.substr(i,j-i)); i=j; }
    std::string real_accum, imag_accum;
    for (const auto& term : terms) {
        if (term.find('I') != std::string::npos) {
            std::string coef = term;
            if (coef == "I" || coef == "+I") coef = "+1"; else if (coef == "-I") coef = "-1"; else { if (!coef.empty() && coef.back()=='I') coef.pop_back(); if (!coef.empty() && coef.back()=='*') coef.pop_back(); }
            if (!imag_accum.empty() && coef[0] != '+' && coef[0] != '-') imag_accum += "+"; imag_accum += coef;
        } else {
            if (!real_accum.empty() && term[0] != '+' && term[0] != '-') real_accum += "+"; real_accum += term;
        }
    }
    Quadratic re = real_accum.empty()? Quadratic{R(0),R(0),2} : parse_quadratic_real(real_accum);
    Quadratic im = imag_accum.empty()? Quadratic{R(0),R(0),2} : parse_quadratic_real(imag_accum);
    return ComplexQuadratic{ re, im };
}

static void print_rational(const Rational &r) {
    if (r.den == 1) std::cout << r.num; else std::cout << r.num << "/" << r.den;
}
static void print_quadratic(const Quadratic &q) {
    bool first = true;
    if (!rational_is_zero(q.a)) { print_rational(q.a); first = false; }
    if (!rational_is_zero(q.b)) {
        if (!first) std::cout << (q.b.num < 0 ? " - " : " + ");
        Rational babs{ std::llabs(q.b.num), q.b.den };
        if (!(babs.num == 1 && babs.den == 1)) { print_rational(babs); std::cout << "*"; }
        else if (first && q.b.num < 0) { std::cout << "-"; }
        std::cout << "sqrt(" << q.rad << ")";
        first = false;
    }
    if (first) std::cout << 0;
}
static void print_cq(const ComplexQuadratic &z) {
    bool printed = false;
    if (!quadratic_is_zero(z.re)) { print_quadratic(z.re); printed = true; }
    if (!quadratic_is_zero(z.im)) {
        if (printed) std::cout << (z.im.a.num < 0 || z.im.b.num < 0 ? " - " : " + ");
        // print |im| with sign handled above for display; this is simplistic but readable
        ComplexQuadratic iim = z; if (z.im.a.num < 0) iim.im.a.num = -iim.im.a.num; if (z.im.b.num < 0) iim.im.b.num = -iim.im.b.num;
        print_quadratic(iim.im); std::cout << "*I";
    }
    if (!printed && quadratic_is_zero(z.im)) std::cout << 0;
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " a1 a2 a3 b1 b2 b3\n";
        std::cerr << "Each ai/bi can include rationals, sqrt(n), and I (e.g., \"-1/2*sqrt(2)-I\").\n";
        return 1;
    }
    std::array<ComplexQuadratic,3> a{ parse_cq_component(argv[1]), parse_cq_component(argv[2]), parse_cq_component(argv[3]) };
    std::array<ComplexQuadratic,3> b{ parse_cq_component(argv[4]), parse_cq_component(argv[5]), parse_cq_component(argv[6]) };

    ComplexQuadratic dot = dot_complex_cq(a, b);
    auto cross = cross_complex_cq(a, b);

    std::cout << "Hermitian dot: "; print_cq(dot); std::cout << "\n";
    std::cout << "Hermitian cross:\n";
    std::cout << "  x = "; print_cq(cross[0]); std::cout << "\n";
    std::cout << "  y = "; print_cq(cross[1]); std::cout << "\n";
    std::cout << "  z = "; print_cq(cross[2]); std::cout << "\n";
    std::cout << "Orthogonal? " << (cq_is_zero(dot) ? "yes" : "no") << "\n";
    return 0;
}


