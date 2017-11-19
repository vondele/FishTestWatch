#!/usr/bin/python

#############################################################################
# Copyright (c) 2017 "Michel Van den Bergh,"
#
# "sprt_elo" is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#############################################################################

from __future__ import division
import math,sys
import argparse

# Below is (slightly modified) code from pyroots.
# Pyroots has its own license which is given below.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# file pyroots/brent.py
#
#############################################################################
# Copyright (c) 2013 by Panagiotis Mavrogiorgos
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name(s) of the copyright holders nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#############################################################################
#
# @license: http://opensource.org/licenses/BSD-3-Clause
# @authors: see AUTHORS.txt


"""
Brent's algorithm for root finding.
"""

EPS = sys.float_info.epsilon

def nearly_equal(a, b, epsilon):
    if a == b:
        return True                         # shortcut. Handles infinities etc
    diff = abs(a - b)
    max_ab = max(abs(a), abs(b), 1)
    if max_ab >= diff or max_ab > 1:
        return diff <= epsilon              # absolute error
    else:
        return diff < epsilon * max_ab      # relative  error

class result(dict):
    def __init__(self, x0, fx0, iterations, func_evaluations, converged, msg=""):
        self['x0'] = x0
        self['fx0'] = fx0
        self['iterations'] = iterations
        self['func_calls'] = func_evaluations
        self['converged'] = converged
        self['msg'] = msg
        
def _extrapolate(fcur, fpre, fblk, dpre, dblk):
    return -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))

def brentq(f, xa, xb, xtol=EPS, epsilon=1e-6, max_iter=500):    
    # initialize counters
    i = 0
    fcalls = 0

    # rename variables in order to be consistent with scipy's code.
    xpre, xcur = xa, xb
    xblk, fblk, spre, scur = 0, 0, 0, 0

    #check that the bracket's interval is sufficiently big.
    if nearly_equal(xa, xb, xtol):
        return result(None, None, i, fcalls, False, "small bracket")

    # check lower bound
    fpre = f(xpre)             # First function call
    fcalls += 1
    if nearly_equal(0,fpre,epsilon):
        return result(xpre, fpre, i, fcalls, True, "lower bracket")

    # check upper bound
    fcur = f(xcur)             # Second function call
    fcalls += 1
    # self._debug(i, fcalls, xpre, xcur, fpre, fcur)
    if nearly_equal(0,fcur,epsilon):
        return result(xcur, fcur, i, fcalls, True, "upper bracket")

    # check if the root is bracketed.
    if fpre * fcur > 0.0:
        return result(None, None, i, fcalls, False, "no bracket")

    # start iterations
    for i in range(max_iter):
        if (fpre*fcur < 0):
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre

        if (abs(fblk) < abs(fcur)):
            xpre = xcur
            xcur = xblk
            xblk = xpre
            fpre = fcur
            fcur = fblk
            fblk = fpre

        # check bracket
        sbis = (xblk - xcur) / 2;
        if abs(sbis) < xtol:
            return result(xcur, fcur, i + 1, fcalls, False, "small bracket")

        # calculate short step
        if abs(spre) > xtol and abs(fcur) < abs(fpre):
            if xpre == xblk:
                # interpolate
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                # extrapolate
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = _extrapolate(fcur, fpre, fblk, dpre, dblk)

            # check short step
            if (2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - xtol)):
                # good short step
                spre = scur
                scur = stry
            else:
                # bisect
                spre = sbis
                scur = sbis
        else:
            # bisect
            spre = sbis
            scur = sbis

        xpre = xcur;
        fpre = fcur;
        if (abs(scur) > xtol):
            xcur += scur
        else:
            xcur += xtol if (sbis > 0) else -xtol

        fcur = f(xcur)     # function evaluation
        fcalls += 1
        # self._debug(i + 1, fcalls, xpre, xcur, fpre, fcur)
        if nearly_equal(0,fcur,epsilon):
            return result(xcur, fcur, i, fcalls, True, "convergence")

    return result(xcur, fcur, i + 1, fcalls, False, "iterations")

# End code from pyroots.

# Taken from https://www.johndcook.com/blog/python_phi/
def Phi(x):
    """
Cumulative standard normal distribution.
"""
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # Save the sign of x
    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)/math.sqrt(2.0)

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)

    return 0.5*(1.0 + sign*y)

def U(n,gamma,A,y):
    """ 
This is a primitive function of e^(gamma y)sin ((n pi y)/A),
multiplied by 2/A*exp(-gamma*y).
"""
    return (2*A*gamma*math.sin(math.pi*n*y/A) - 2*math.pi*n*math.cos (math.pi*n*y/A))/(A**2*gamma**2 + math.pi**2 * n**2)

class Brownian:

    def __init__(self,a=-1.0,b=1.0,mu=0.0,sigma=0.005):
        self.a=a
        self.b=b
        self.mu=mu
        self.sigma=sigma
        self.sigma2=sigma**2
        assert self.sigma2>0

    def outcome_cdf(self,T=None,y=None):
        # In case of slow convergence use Siegmund approximation.
        # Otherwise use exact formula.
        sigma2=self.sigma2
        mu=self.mu
        gamma=mu/sigma2
        A=self.b-self.a
        if sigma2*T/A**2<1e-2 or abs(gamma*A)>15:
            ret=self.outcome_cdf_alt2(T,y)
        else:
            ret=self.outcome_cdf_alt1(T,y)
        assert -1e-3 <= ret <= 1+1e-3
        return ret
        
    def outcome_cdf_alt1(self,T=None,y=None):
        """ 
Computes the probability that the particle passes to the
right of (T,y), the time axis being vertically oriented.
This may give a numerical exception if math.pi**2*sigma2*T/(2*A**2) 
is small.
"""
        mu=self.mu
        sigma2=self.sigma2
        A=self.b-self.a
        x=0-self.a
        y=y-self.a
        gamma=mu/sigma2
        n=1
        s=0.0
        lambda_1=((math.pi/A)**2)*sigma2/2+(mu**2/sigma2)/2
        t0=math.exp(-lambda_1*T-x*gamma+y*gamma)
        while True:
            lambda_n=((n*math.pi/A)**2)*sigma2/2+(mu**2/sigma2)/2
            t1=math.exp(-(lambda_n-lambda_1)*T)
            t3=U(n,gamma,A,y)
            t4=math.sin(n*math.pi*x/A)
            s+=t1*t3*t4
            if abs(t0*t1*t3)<=1e-9:
                break
            n+=1
        if gamma*A>30:     # avoid numerical overflow
            pre=math.exp(-2*gamma*x)
        elif abs(gamma*A)<1e-8: # avoid division by zero
            pre=(A-x)/A
        else:
            pre=(1-math.exp(2*gamma*(A-x)))/(1-math.exp(2*gamma*A))
        return pre+t0*s

    def outcome_cdf_alt2(self,T=None,y=None):
        """
Siegmund's approximation. We use it as backup if our
exact formula converges too slowly. To make the evaluation
robust we use the asymptotic development of Phi.
"""
        denom=math.sqrt(T*self.sigma2)
        offset=self.mu*T
        gamma=self.mu/self.sigma2
        a=self.a
        b=self.b
        z=(y-offset)/denom
        za=(-y+offset+2*a)/denom
        zb=(y-offset-2*b)/denom
        t1=Phi(z)
        if gamma*a>=5:
            t2=-math.exp(-za**2/2+2*gamma*a)/math.sqrt(2*math.pi)*(1/za-1/za**3)
        else:
            t2=math.exp(2*gamma*a)*Phi(za)
        if gamma*b>=5:
            t3=-math.exp(-zb**2/2+2*gamma*b)/math.sqrt(2*math.pi)*(1/zb-1/zb**3)
        else:
            t3=math.exp(2*gamma*b)*Phi(zb)
        return t1+t2-t3

def scale(de):
    return (4*10**(-de/400))/(1+10**(-de/400))**2

def wdl(elo,de):
    w=1/(1+10**((-elo+de)/400))
    l=1/(1+10**((elo+de)/400))
    d=1-w-l
    return(w,d,l)

def draw_elo_calc(draw_ratio):
    return 400*(math.log(1/((1-draw_ratio)/2.0)-1)/math.log(10))

BayesElo=0
LogisticElo=1

class sprt:
    def __init__(self,alpha=0.05,beta=0.05,elo0=0,elo1=5,mode=BayesElo):
        self.elo0_raw=elo0
        self.elo1_raw=elo1
        self.a=math.log(beta/(1-alpha))
        self.b=math.log((1-beta)/alpha)
        self.mode=mode

    def set_state(self,W=None,D=None,L=None):
        self.N=W+D+L
        self.dr=D/self.N
        self.de=draw_elo_calc(self.dr)
        if self.mode==LogisticElo:
            sf=scale(self.de)
            self.elo0=self.elo0_raw/sf
            self.elo1=self.elo1_raw/sf
        else:
            self.elo0=self.elo0_raw;
            self.elo1=self.elo1_raw;
        w0,d0,l0=wdl(self.elo0,self.de)
        w1,d1,l1=wdl(self.elo1,self.de)
        self.llr_win=math.log(w1/w0)
        self.llr_draw=math.log(d1/d0)
        self.llr_loss=math.log(l1/l0)
        self.llr=W*self.llr_win+D*self.llr_draw+L*self.llr_loss
        self.llr_raw=self.llr
# record if llr is outside legal range
        self.clamped=False
        if self.llr<self.a+self.llr_loss or self.llr>self.b+self.llr_win:
            self.clamped=True
# now normalize llr (if llr is not legal then the implications of this are unclear)
        slope=self.llr/self.N
        if self.llr<self.a:
            self.T=self.a/slope
            self.llr=self.a
        elif self.llr>self.b:
            self.T=self.b/slope
            self.llr=self.b
        else:
            self.T=self.N

    def outcome_prob(self,elo):
        """
The probability of a test with the given elo with worse outcome
(faster fail, slower pass or a pass changed into a fail).
"""
        w,d,l=wdl(elo,self.de)
        mu=w*self.llr_win+d*self.llr_draw+l*self.llr_loss
        mu2=w*self.llr_win**2+d*self.llr_draw**2+l*self.llr_loss**2
        sigma2=mu2-mu**2
        sigma=math.sqrt(sigma2)
        return Brownian(a=self.a,b=self.b,mu=mu,sigma=sigma).outcome_cdf(T=self.T,y=self.llr)

    def lower_cb(self,p):
        """
Maximal elo value such that the observed outcome of the test has probability
less than p.
"""
        avg_elo=self.elo0+self.elo1
        delta=self.elo1-self.elo0
        N=30
# Various error conditions must be handled better here!
        while True:
            Elo0=avg_elo-N*delta
            Elo1=avg_elo+N*delta
            sol=brentq(lambda elo:self.outcome_prob(elo)-(1-p),Elo0,Elo1)
            if sol['msg']=='no bracket':
                N*=2
                continue
            break
        return sol['x0']

    def analytics(self,p=0.05):
        ret={}
        ret['LLR']=self.llr_raw
        sf=scale(self.de)
        ret['elo0']=self.elo0*sf
        ret['elo1']=self.elo1*sf
        ret['elo']=self.lower_cb(0.5)*sf
        ret['ci']=[self.lower_cb(p/2)*sf,self.lower_cb(1-p/2)*sf]
        ret['LOS']=self.outcome_prob(0)
        ret['H0_reject']=self.outcome_prob(self.elo0)
        ret['H1_reject']=1-self.outcome_prob(self.elo1)
        ret['draw_elo']=self.de
        ret['draw_ratio']=self.dr
        ret['games']=self.N
        ret['clamped']=self.clamped
        ret['a']=self.a
        ret['b']=self.b
        return ret

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha",help="probability of a false positve",type=float,default=0.05)
    parser.add_argument("--beta" ,help="probability of a false negative",type=float,default=0.05)
    parser.add_argument("--logistic", help="Express input in logistic elo",
                        action='store_true')
    parser.add_argument("--elo0", help="H0 (expressed in BayesElo)",type=float,default=0.0)
    parser.add_argument("--elo1", help="H1 (expressed in BayseElo)",type=float,default=5.0)
    parser.add_argument("--level",help="confidence level",type=float,default=0.95)
    parser.add_argument("-W", help="number of won games",type=int,required=True)
    parser.add_argument("-D", help="number of draw games",type=int,required=True)
    parser.add_argument("-L", help="nummer of lost games",type=int,required=True)
    args=parser.parse_args()
    W,D,L=args.W,args.D,args.L
    alpha=args.alpha
    beta=args.beta
    elo0=args.elo0
    elo1=args.elo1
    level=args.level
    # sanitize
    W=max(W,0.001)
    D=max(D,0.001)
    L=max(L,0.001)
    elo0=max(elo0,-100.0)
    elo1=min(elo1,100.0)
    elo1=max(elo1,elo0+0.1)
    alpha=min(alpha,0.9)
    alpha=max(alpha,1e-3)
    beta=min(beta,0.9)
    beta=max(beta,1e-3)
    level=max(level,0.001)
    level=min(level,0.999)
    mode=LogisticElo if args.logistic else BayesElo
    s=sprt(alpha=alpha,beta=beta,elo0=elo0,elo1=elo1,mode=mode)
    s.set_state(W,D,L)
    p=1-level
    a=s.analytics(p)
    print("Design parameters")
    print("=================")
    print("False positives             :  %4.1f%%" % (100*alpha,))
    print("False negatives             :  %4.1f%%" % (100*beta,))
    print("[Elo0,Elo1]                 :  [%.1f,%.1f] %s" % (elo0,elo1,"" if mode==LogisticElo else "(BayesElo)"))
    print("Confidence level            :  %4.1f%%" % (100*(1-p),))
    print("Estimates")
    print("=========")
    print("Elo                         :  %.1f"    % a['elo'])
    print("Confidence interval         :  [%.1f,%.1f] (%4.1f%%)"  % (a['ci'][0],a['ci'][1],100*(1-p)))
    print("Confidence of gain ('LOS')  :  %4.1f%%" % (100*a['LOS'],))
    print("Context")
    print("=======")
    print("Games                       :  %d" % (a['games'],))
    print("Draw ratio                  :  %4.1f%%"    % (100*a['draw_ratio'],))
    print("DrawElo                     :  %.1f (BayesElo)"    % a['draw_elo'])
    print("LLR [u,l]                   :  %.2f %s [%.2f,%.2f]"       % (a['LLR'], '(clamped)' if a['clamped'] else '',a['a'],a['b']))
    print("[H0,H1] rejection           :  [%4.1f%%,%4.1f%%]" % (100*a['H0_reject'],100*a['H1_reject']))
    print("[Elo0,Elo1]                 :  [%.2f,%.2f]"    % (a['elo0'],a['elo1']))


