# pySVM
Lab 1 of course VE445@UM-SJTU JI. The implementation of SVM, and supports LSVM for online training, and Kernel SVM and Soft Margin SVM.


Require python 3.5+ and cvxopt 1.1.9+  
Run the file with:  
    svm = 'SVMMODEL'(sample_file, label_file)  
    #svm.input_margin(margin) # if soft margin svm is required  
    svm.training(kernel=kernel_name, parameter=parameter) # for not LSVM  
    #svm.online_training(kernel=kernel_name, parameter=parameter) #for LSVM  
    print(svm.testing(test_sample, test_label)) # an original test samples and labels are splitted from the origin data set  
                                                # can input extra test samples and labels  


Some notes for LSVM:  
    - On the base of Soft Margin SVM and Kernel SVM, supports online training.  
    - For each batch, stores only the support vectors, and throw away all other vectors.  
    - Former support vectors will have a higher constraint, and have higher value in the later training.  
