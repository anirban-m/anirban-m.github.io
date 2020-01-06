# anirban-m.github.io
This page hosts some concept maps and numerical codes for my doctoral project 
                    "__unitary renormalization group(RG) for correlated electronic systems__"
                    
  __Table of Contents__                  
 - [unitary-disentanglement-RG](/unitary-disentanglement-RG)
    + [benchmarking2DHubbard](/unitary-disentanglement-RG/benchmarking2DHubbard)<br>
    In the folder we have placed a ipython notebook that demonstrates how to carry out the RG numerically for the Hubbard model
    by iterating the RG equations. From the fixed point solutions we compute the ground state energy density for zero doping and 12.5 % doping via a finite size scaling plot. The numerical values are in excellent agreement with those presented by the Simon's collaboration [LeBlanc.et.al](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.5.041041).
      + [Main Concept behind the Unitary RG](https://anirban-m.github.io/unitary-disentanglement-RG/benchmarking2DHubbard/preliminaries.html)<br>
        This page gives a introduction to the RG formalism via demonstrating the essential concept of Hamiltonian block diagonalization via unitary transformations. The formalism is presented in complete detail in our arxiv work [arxiv-1802.06528](https://arxiv.org/abs/1802.06528)<br>
       +[GroundStateEnergyDensity](/unitary-disentanglement-RG/benchmarking2DHubbard/GroundStateEnergyDensity.ipynb)<br>
       The ipython notebook demonstrating the numerical visualization of the ground state energy density's finite size scaling for the 2d Hubbard model.<br>
         __To give the motivation and implementation procedure of the entanglement renormalization we present our viewers with a presentation. [Entanglement-Renormalization](https://xp7orer.github.io/entanglement-renormalization/EntanglementRenormalization-InformationTheoreticPerspective.slides.html)__<br>
	 Following this demonstration we describe the various coding aspects of the entanglement renormalization program. 
      + [Entanglement-Distillation](/unitary-disentanglement-RG/EntanglementDistillationRG.ipynb)<br>
          In this ipython notebook we first perform the unitary RG to obtain the updation of Hamiltonian parameters. Following which we write down the ground state wavefunction in a file. The updated parameter list along with the form of the interaction allows us to regenerate the many body states at the earlier RG steps. In this code we have performed 6 reverse RG steps for a system of 28 pseudospins. At each reverse RG step 4 pseudospins get re-entangled to the ground state wavefunction at the fixed point which involves only four pseudospins.   
    +[LossFunction](/unitary-disentanglement-RG/LossFunction.ipynb)<br>
          In this notebook we check the performance of the unitary RG based reconstruction of the wavefunctions in the reverse RG steps. This is done by computing the energy uncertainity of the model Hamiltonian with respect to the earlier step wavefunctions.<br>
    +[Fidelity](/unitary-disentanglement-RG/FidelityScaling.ipynb)<br>
          In this notebook we compute the overlap and its squared overlap between the fixed point wavefunction and reverse RG iterated wavefunction. It shows a significant fall in the reverse RG steps. Indicating the dramatic compression capability of this RG. We later show how even while compression happens the essential information is contained in the many body wavefunction across all the RG transformation steps.
    + [EntanglementGeometry](/unitary-disentanglement-RG/EntanglementGeometry.ipynb)<br>
          In this ipython notebook we compute the Schmidt Coefficients for two kinds of partitions 
          -> pair of degrees of freedom  and the rest N-2 degrees of freedom i.e. [2,N-2]. This allows us to construct the reduced density for 2 pseudospin for all [i,j] pairs in a system of 28 degrees of freedom.<br>
          ->1 degree of freedom  and the rest N-1 degrees of freedom [1,N-1]. This allows us to construct the reduced density matrix for 1 pseudospin in a system of 28 degrees of freedom.<br>
          ->From here we compute the mutual information a measure of pairwise entanglement for all 28C2 pairs of pseudospins<br>
          This data set at every RG step characterizes all the entanglement features upto 2 pseudospins.
     + [DataAnalysisPlotsGeneration](/unitary-disentanglement-RG/DataAnalysisPlotsGeneration.ipynb)<br>
          In this ipython notebook we generate plots that allow us to visualize the entanglement renormalization:
          The quantities that we compute include
          ->The mutual information values for every pair of degree of freedom.<br>
          ->the Schmidt coefficients for the following partitions of degrees of freedoms [1,N-1] and [2,N-2].
          ->The Information geodesic that tracks the strongest entangled pair in a system and the diameter that tracks the weakest pair.<br>
          ->Finally we perform a information theoretic analysis of the RG process itself by computing the mutual information between entanglement feature datasets.<br>
         __Information Theoretic Analysis of the entanglement renormalization framework__
	       We seek to perform a information theoretic analysis of the RG procedure itself. In order to lay the platform for such a analysis we fix the nomenclature.<br> 
		   ->Let us label every pair of degrees of freedom in the entangled state as (i,j).<br>
		   ->Associated with every pair is the feature(F(i,j))-mutual information value F(i,j)=I(i:j) that lies between 0 and 2log2. More the values I(i:j) stronger is the "entangledness" of the pair.<br>
		   ->Based on this feature we define a classifier(C(i,j)) if I(i:j)<log2 then the pair is "weakly" entangled and C(i,j)=0 else the pair is "strongly" entangled C(i,j)=1.<br>
		   ->We define a target classification data with respect to the RG fixed point. In our setup we have 6 RG steps. So we compute two quantities:<br>
		   1>Mutual Information content between the target classification and Feature at every RG setup.<br>
		   2>Mutual Information content between the Feature at initial step and Feature at every later RG transformation step.<br>
		    Here 1 quantifies the amount of information the bottlenecked feature representation carries of the target classification.<br>
		Here 2 quantifies the amount of compression between the initial feature representation and the bottlenecked representation.<br> 
