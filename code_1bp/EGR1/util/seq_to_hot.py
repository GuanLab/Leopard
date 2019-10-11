def seq_to_hot(seq):
    import numpy as np
    seq=seq.replace('a','A')
    seq=seq.replace('c','C')
    seq=seq.replace('g','G')
    seq=seq.replace('t','T')
    seq=seq.replace('n','N')
    Aseq=seq
    Aseq=Aseq.replace('A','1')
    Aseq=Aseq.replace('C','0')
    Aseq=Aseq.replace('G','0')
    Aseq=Aseq.replace('T','0')
    Aseq=Aseq.replace('N','0')
    Aseq=np.asarray(list(Aseq),dtype='float32')
    Cseq=seq
    Cseq=Cseq.replace('A','0')
    Cseq=Cseq.replace('C','1')
    Cseq=Cseq.replace('G','0')
    Cseq=Cseq.replace('T','0')
    Cseq=Cseq.replace('N','0')
    Cseq=np.asarray(list(Cseq),dtype='float32')
    Gseq=seq
    Gseq=Gseq.replace('A','0')
    Gseq=Gseq.replace('C','0')
    Gseq=Gseq.replace('G','1')
    Gseq=Gseq.replace('T','0')
    Gseq=Gseq.replace('N','0')
    Gseq=np.asarray(list(Gseq),dtype='float32')
    Tseq=seq
    Tseq=Tseq.replace('A','0')
    Tseq=Tseq.replace('C','0')
    Tseq=Tseq.replace('G','0')
    Tseq=Tseq.replace('T','1')
    Tseq=Tseq.replace('N','0')
    Tseq=np.asarray(list(Tseq),dtype='float32')
    hot=np.vstack((Aseq,Cseq,Gseq,Tseq))
    return hot
