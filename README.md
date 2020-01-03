# xp7orer.github.io
This page hosts some concept maps and numerical codes for my doctoral project 
                    "__unitary renormalization group(RG) for correlated electronic systems__"
                    
  __Table of Contents__                  
 - [unitary-disentanglement-RG](/unitary-disentanglement-RG)
    + [benchmarking2DHubbard](/unitary-disentanglement-RG/benchmarking2DHubbard)<br>
    In the folder we have placed a ipython notebook that demonstrates how to carry out the RG numerically for the Hubbard model
    by iterating the RG equations. From the fixed point solutions we compute the ground state energy density for zero doping and 12.5 % doping via a finite size scaling plot. The numerical values are in excellent agreement with those presented by the Simon's collaboration [LeBlanc.et.al](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.5.041041).
      + [Main Concept behind the Unitary RG](https://xp7orer.github.io/unitary-disentanglement-RG/benchmarking2DHubbard/preliminaries.html)<br>
        This page gives a introduction to the RG formalism via demonstrating the essential concept of Hamiltonian block diagonalization via unitary transformations. The formalism is presented in complete detail in our arxiv work [arxiv-1802.06528](https://arxiv.org/abs/1802.06528)<br>
       +[GroundStateEnergyDensity](/unitary-disentanglement-RG/benchmarking2DHubbard/GroundStateEnergyDensity.ipynb)<br>
       The ipython notebook demonstrating the numerical visualization of the ground state energy density's finite size scaling for the 2d Hubbard model.
        + [Entanglement-Distillation](/unitary-disentanglement-RG/EntanglementDistillationRG.ipynb)<br>
          In this ipython notebook we first perform the unitary RG to obtain the updation of Hamiltonian parameters. Following which we write down the ground state wavefunction in a file. The updated parameter list along with the form of the interaction allows us to regenerate the many body states at the earlier RG steps. In this code we have performed 6 reverse RG steps for a system of 28 pseudospins. At each reverse RG step 4 pseudospins get re-entangled to the ground state wavefunction at the fixed point which involves only four pseudospins.   
