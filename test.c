#include "c_api.h"
#include "stdio.h"


int main(int argc, char **argv) {
    DN_InitializeFromArgs(argc, argv, false);
    
    // Define model parameter collection
    DN_ParameterCollection* pc = DN_NewParameterCollection();

    // Define trainer on the pc defined above
    DN_SimpleSGDTrainer* trainer = DN_NewSimpleSGDTrainer(pc, 0.1);

    // Build computation graph
    DN_ComputationGraph* cg = DN_NewComputationGraph();

    long d[] ={1, 3};
    DN_Dim* dim = DN_NewDimFromArray(d, 2, 1);
    //char *name = "W";
    DN_Parameter* W_P = DN_AddParametersToCollection(pc, dim, "W");
    //DN_PrintTensor(DN_ParameterValues(W_P));
    DN_Expression* W = DN_LoadParamToCG(cg, W_P);

    float x_values[3] = {0.5, 0.3, 0.7};
    long x_d[] = {3};
    DN_Dim* x_dim = DN_NewDimFromArray(x_d, 1, 1);
    DN_Expression* x = DN_AddInputToCG(cg, x_dim, x_values, 3);

    float y_value[1];
    long y_d[] = {1};
    DN_Dim* y_dim = DN_NewDimFromArray(y_d, 1, 1);
    DN_Expression* y = DN_AddInputToCG(cg, y_dim, y_value, 1);

    DN_Expression* w_x = DN_Multiply(W, x);
    DN_Expression* y_pred = DN_Logistic(w_x);

    DN_Expression* l = DN_BinaryLogLoss(y_pred, y);
    
    DN_PrintGraphviz(cg);

    // Set inputs' values now. Can not be done now :( Known Bugs!
    // x_values[0] = 0.5; x_values[1] = 0.3;  x_values[2] = 0.7;
    // y_value[0] = 1.0;

    DN_PrintTensor(DN_GetExprValue(y));
    // Compute forward now
    float loss = DN_Forward(cg, l);
    printf("%f\n", loss);

    // Compute backward now
    DN_Backward(cg, l, false);
    DN_SimpleSGDUpdate(trainer);

    //printf("W:");
    //DN_PrintTensor(DN_ParameterValues(W_P));

    // Compute forward again
    loss = DN_Forward(cg, l);
    printf("%f\n", loss);

    return 0;
}
