#include "c_api.h"
#include "stdio.h"


int main(int argc, char **argv) {
    DN_InitializeFromArgs(argc, argv, false);
    
    // Define model parameter collection
    DN_ParameterCollection* pc = DN_NewParameterCollection();

    // Define parameters
    long d[] ={1, 3};
    DN_Dim* dim = DN_NewDimFromArray(d, 2, 1);
    DN_ParameterInitGlorot* init = DN_NewParameterInitGlorot(false, 1.0);
    DN_Parameter* W_P = DN_AddParametersToCollectionGlorot(pc, dim, init, "W");

    // Define trainer on the pc defined above
    DN_SimpleSGDTrainer* trainer = DN_NewSimpleSGDTrainer(pc, 0.1);

    // Build computation graph for each example
    const unsigned ITERATIONS = 30;
    for (unsigned iter = 0; iter < ITERATIONS; ++iter) {
        
        DN_ComputationGraph* cg = DN_NewComputationGraph();
        DN_Expression* W = DN_LoadParamToCG(cg, W_P);
        
        for (unsigned mi = 0; mi < 4; ++mi) {
            bool x1 = mi % 2;
            bool x2 = (mi / 2) % 2;
            float x_values[3];
            x_values[0] = x1 ? 1 : -1;
            x_values[1] = x2 ? 1 : -1;
            long x_d[] = {3};
            DN_Dim* x_dim = DN_NewDimFromArray(x_d, 1, 1);
            DN_Expression* x = DN_AddInputToCG(cg, x_dim, x_values, 3);

            float y_value[1];
            y_value[0] = (x1 != x2) ? 1 : -1;
            long y_d[] = {1};
            DN_Dim* y_dim = DN_NewDimFromArray(y_d, 1, 1);
            DN_Expression* y = DN_AddInputToCG(cg, y_dim, y_value, 1);

            //y = logistic(w*x)
            DN_Expression* w_x = DN_Multiply(W, x);
            DN_Expression* y_pred = DN_Logistic(w_x);
            
            DN_Expression* l = DN_BinaryLogLoss(y_pred, y);

            // Compute forward now
            float loss = DN_Forward(cg, l);
            printf("%f\n", loss);

            // Compute backward now
            DN_Backward(cg, l, false);
            DN_SimpleSGDTrainerUpdate(trainer);
        }

        // Clean ups
        
        DN_DeleteComputationGraph(cg);
        
    }
    DN_DeleteSimpleSGDTrainer(trainer);
    DN_DeleteParameter(W_P);
    DN_DeleteParameterInitGlorot(init);
    DN_DeleteDim(dim);
    DN_DeleteParameterCollection(pc);
    return 0;
}
