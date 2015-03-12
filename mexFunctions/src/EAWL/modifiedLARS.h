/*!
 *
 *                File modifiedLARS.h
 *                by Sebastian Hitziger
 *                (modified from Julien Mairal)
 * 
 * Contains function modLARS: modified version of least angle regression (LARS)
 * for shift invariant dictionaries. Ensures that at most one translate of each
 * shiftable atom is activated.
 *
 * Used in epoched adaptive waveform learning (EAWL.h)
 *
 * The current implementation is a modification of the LARS algorithm from
 * the SPAMS software package by Julien Mairal under the 
 * GNU General Public License, see <http://www.gnu.org/licenses/>.
 *
 * */

#ifndef DECOMP_JITTER_H
#define DECOMP_JITTER_H

#include <utils.h>


/// Auxiliary function for lasso % TODO : recode avec matrix inversion lemma
template <typename T>
void modLARS( Vector<T>& DtR,               //KS 
                const CovShMat<T>& DtD,     //KxKxn2 
                Matrix<T>& Gs,
                Matrix<T>& Ga,              //KxL
                Matrix<T>& invGs,
                Vector<T>& u,
                Vector<T>& coeffs,
                Vector<int>& ind,
                Vector<T>& work1,           // KS
                Vector<T>& work2,           // KS
                Vector<T>& work3,           // KS
                T& normX,
                const T constraint,
                const bool pos,
                int length_path = -1) {

        
    const int K = DtD.K();
    const int n1 = DtD.n1();
    const int n2 = DtD.n2();
    const int S = n1+1;
    const int KK = S*K;
    const int L = Gs.n();

    if (length_path < 0) length_path=4*L;


    // get pointers for direct access
    T* const pr_Gs       = Gs.rawX();
    T* const pr_invGs    = invGs.rawX();
    T* const pr_Ga       = Ga.rawX();
    T* const pr_work1    = work1.rawX();
    T* const pr_work2    = work2.rawX();
    T* const pr_work3    = work3.rawX();
    T* const pr_u        = u.rawX();
    T* const pr_DtR      = DtR.rawX();
    T* const pr_coeffs   = coeffs.rawX();
    int* const pr_ind    = ind.rawX();


    // declarations
    bool newAtom=true;
    int currentInd,i,ik,is,j,k,l,first_zero,index,index2;
    int iter=0;
    T coeff1,coeff2,coeff3,step,step_max2,thrs = 0;

    // initializations 
    coeffs.setZeros();
    ind.set(-1);
    
    // Find the most correlated element
    currentInd = pos ? DtR.max() : DtR.fmax();
    if (abs(DtR[currentInd]) < constraint) return;


    // loop until sparsity is reached
    for (i = 0; i<L; ++i) {
        ++iter;
        if (newAtom) {
            ik = currentInd/S;  // indicates atom
            is = currentInd%S;  // indicates its position
            pr_ind[i] = currentInd;
            
            //TODO: efficient
            DtD.generateGk(ik,is,pr_Ga+i*KK);
            for (j = 0; j<=i; ++j) {
                pr_Gs[i*L+j]=pr_Ga[i*KK+pr_ind[j]];
            }
            // Update inverse of Gs using matrix inversion lemma with schur
            // complement 
            if (i == 0) {
                pr_invGs[0]=T(1.0)/pr_Gs[0];
            } else {
                cblas_symv<T>(CblasColMajor,CblasUpper,i,T(1.0),
                        pr_invGs,L,pr_Gs+i*L,1,T(0.0),pr_u,1);
                T schur =
                        T(1.0)/(pr_Gs[i*L+i]-cblas_dot<T>(i,pr_u,1,pr_Gs+i*L,1));
                pr_invGs[i*L+i]=schur;
                cblas_copy<T>(i,pr_u,1,pr_invGs+i*L,1);
                cblas_scal<T>(i,-schur,pr_invGs+i*L,1);
                cblas_syr<T>(CblasColMajor,CblasUpper,i,schur,pr_u,1,
                      pr_invGs,L);
            }
        }
      
        // Compute the path direction epsilon, then calculate u as invGs * eps_Lambda
        for (j = 0; j<=i; ++j)
            pr_work1[j]= pr_DtR[pr_ind[j]] > 0 ? T(1.0) : T(-1.0); // work corresponds to epsilon (?!)
        cblas_symv<T>(CblasColMajor,CblasUpper,i+1,T(1.0),pr_invGs,L,
            pr_work1,1,T(0.0),pr_u,1);

        // Compute the step on the path, first in case of active becoming inactive
        T step_max = INFINITY; 
        first_zero = -1;
        for (j = 0; j<=i; ++j) {
            T ratio = -pr_coeffs[j]/pr_u[j];
            if (ratio > 0 && ratio <= step_max) {
                step_max=ratio;
                first_zero=j;
            }
        }

        T current_correlation = abs<T>(pr_DtR[pr_ind[0]]);


        //TODO: too expensive, only needs to be performed for unblocked elements
        cblas_gemv<T>(CblasColMajor,CblasNoTrans,KK,i+1,T(1.0),pr_Ga,
                KK,pr_u,1,T(0.0),pr_work1,1);
        cblas_copy<T>(KK,pr_work1,1,pr_work2,1);
        cblas_copy<T>(KK,pr_work1,1,pr_work3,1);

        // prevent atoms from activation if one of their translates is already
        // active 
        for (j = 0; j<=i; ++j) {
            for (k = 0; k<S; ++k) {
                pr_work1[(pr_ind[j]/S)*S+k]=INFINITY;
                pr_work2[(pr_ind[j]/S)*S+k]=INFINITY;
            }
        } 
        for (j = 0; j<KK; ++j) {
            pr_work1[j] = ((pr_work1[j] < INFINITY) && (pr_work1[j] > T(-1.0))) ? 
                (pr_DtR[j]+current_correlation)/(T(1.0)+pr_work1[j]) : INFINITY;
        }
        for (j = 0; j<KK; ++j) {
            pr_work2[j] = ((pr_work2[j] < INFINITY) && (pr_work2[j] < T(1.0))) ? 
                (current_correlation-pr_DtR[j])/(T(1.0)-pr_work2[j]) : INFINITY;
        }
        // enforce positivity constraint
        if (pos) for (j = 0; j<KK; ++j) pr_work1[j]=INFINITY;

        // find smallest stepsize
        index = cblas_iamin<T>(KK,pr_work1,1);
        index2 = cblas_iamin<T>(KK,pr_work2,1);
        index = abs(pr_work1[index])<abs(pr_work2[index2]) ? index : index2; 
        step = MIN(abs(pr_work1[index]),abs(pr_work2[index]));
        //if (step <1e-10) {
        //    mexPrintf("small step");
        //    mexPrintf("Work 1, index %d: %g\n",index,pr_work1[index]);
        //    mexPrintf("Work 2, index %d: %g\n",index,pr_work2[index]);
        //}

        // Choose next element
        currentInd = index;

        // compute the coefficients of the polynome representing normX^2
        coeff1 = 0;
        for (j = 0; j<=i; ++j)
            coeff1 += pr_DtR[pr_ind[j]] > 0 ? pr_u[j] : -pr_u[j];
        coeff2 = 0;
        for (j = 0; j<=i; ++j)
            coeff2 += pr_DtR[pr_ind[j]]*pr_u[j];
        coeff3 = normX-constraint;

        step_max2 = current_correlation-constraint;

        step = MIN(MIN(step,step_max2),step_max);
        if (step == INFINITY) {
            break; // stop the path
        }
        if (i==L-1) {
            step = step_max2;
        }

        cblas_axpy<T>(i+1,step,pr_u,1,pr_coeffs,1);
        if (pos) {
            for (j = 0; j<=i; ++j)
                if (pr_coeffs[j] < 0) pr_coeffs[j]=0;
        }

        // Update correlations
        cblas_axpy<T>(KK,-step,pr_work3,1,pr_DtR,1);

        // Update normX
        normX += coeff1*step*step-2*coeff2*step;

        // Update norm1
        thrs += step*coeff1;

        // Choose next action: deactivation vs. activation
        if (step == step_max) {     // first case: deactivation of coefficient
            // Downdate, remove first_zero
            // Downdate Ga, Gs, invGs, ind, coeffs
            for (int j = first_zero; j<i; ++j) {
                cblas_copy<T>(KK,pr_Ga+(j+1)*KK,1,pr_Ga+j*KK,1);
                pr_ind[j]=pr_ind[j+1];
                pr_coeffs[j]=pr_coeffs[j+1];
            }
            pr_ind[i]=-1;
            pr_coeffs[i]=0;
            for (int j = first_zero; j<i; ++j) {
                cblas_copy<T>(first_zero,pr_Gs+(j+1)*L,1,pr_Gs+j*L,1);
                cblas_copy<T>(i-first_zero,pr_Gs+(j+1)*L+first_zero+1,1,
                    pr_Gs+j*L+first_zero,1);
            }
            const T schur = pr_invGs[first_zero*L+first_zero];
            cblas_copy<T>(first_zero,pr_invGs+first_zero*L,1,pr_u,1);
            cblas_copy<T>(i-first_zero,pr_invGs+(first_zero+1)*L+first_zero,L,
                pr_u+first_zero,1);
            for (int j = first_zero; j<i; ++j) {
                cblas_copy<T>(first_zero,pr_invGs+(j+1)*L,1,pr_invGs+j*L,1);
                cblas_copy<T>(i-first_zero,pr_invGs+(j+1)*L+first_zero+1,1,
                    pr_invGs+j*L+first_zero,1);
            }
            cblas_syr<T>(CblasColMajor,CblasUpper,i,T(-1.0)/schur, pr_u,1,pr_invGs,L);
            newAtom=false;
            i=i-2;
            // TODO remove atoms from prohibited list (all translates of removed
            // atom)
      } else {      // second case: activation of new atom
         newAtom=true;
      }
      if (iter >= length_path || abs(step) < 1e-15 ||
            step == step_max2 || (normX < 1e-15) ||
            (i == (L-1))
         )  {
         break;
      }
   }
}

#endif 
