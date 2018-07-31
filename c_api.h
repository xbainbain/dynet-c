#ifndef DYNET_C_C_API_H_
#define DYNET_C_C_API_H_

// -----------------------------------------------------------------------------
// C API for dynet.

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

// -----------------------------------------------------------------------------
// init.h

typedef struct DN_DynetParams DN_DynetParams;

DN_DynetParams* DN_NewDynetParams();
void DN_DeleteDynetParams(DN_DynetParams* dp);

void DN_InitializeFromParams(DN_DynetParams* dp);
void DN_InitializeFromArgs(int argc, char** argv, bool shared_parameters);
void DN_Cleanup();
void DN_ResetRng(unsigned int seed);


// -----------------------------------------------------------------------------
// param-init.h
typedef struct DN_ParameterInitNormal DN_ParameterInitNormal;
typedef struct DN_ParameterInitUniform DN_ParameterInitUniform;
typedef struct DN_ParameterInitConst DN_ParameterInitConst;
typedef struct DN_ParameterInitIdentity DN_ParameterInitIdentity;
typedef struct DN_ParameterInitGlorot DN_ParameterInitGlorot;
typedef struct DN_ParameterInitSaxe DN_ParameterInitSaxe;
typedef struct DN_ParameterInitFromFile DN_ParameterInitFromFile;
typedef struct DN_ParameterInitFromArray  DN_ParameterInitFromArray;

DN_ParameterInitNormal* DN_NewParameterInitNormal(float m, float v);
DN_ParameterInitUniform* DN_NewParameterInitUniform(float l, float r);
DN_ParameterInitConst* DN_NewParameterInitConst(float c);
DN_ParameterInitIdentity* DN_NewParameterInitIdentity();
DN_ParameterInitGlorot* DN_NewParameterInitGlorot(bool is_lookup, float gain);
DN_ParameterInitSaxe* DN_NewParameterInitSaxe(float gain);
DN_ParameterInitFromFile* DN_NewParameterInitFromFile(const char* f);
DN_ParameterInitFromArray* DN_NewParameterInitFromArray(const float* a,
                                                        unsigned int n);

// -----------------------------------------------------------------------------
// dim.h
typedef struct DN_Dim DN_Dim;

DN_Dim* DN_NewDim();
DN_Dim* DN_NewDimFromArray(const long* a, unsigned int nd, unsigned int b);
void DN_DeleteDim(DN_Dim* dim);

// -----------------------------------------------------------------------------
// tensor.h
typedef struct DN_Tensor DN_Tensor;

//DN_Tensor* DN_NewTensor();
void DN_DeleteTensor(DN_Tensor* t);

DN_Dim* DN_TensorDim(DN_Tensor* t);
float* DN_TensorToArray(DN_Tensor* t);
float DN_TensorToScalar(DN_Tensor* t);
float* DN_TensorToScaleArray(DN_Tensor* t);
void DN_PrintTensor(DN_Tensor* t);

// -----------------------------------------------------------------------------
// model.h
typedef struct DN_Parameter DN_Parameter;
typedef struct DN_LookupParameter DN_LookupParameter;
typedef struct DN_ParameterCollection DN_ParameterCollection;

// **Parameter**
DN_Parameter* DN_NewParameter();
void DN_DeleteParameter(DN_Parameter* p);
const char* DN_GetParameterFullName(DN_Parameter* p);
void DN_ZeroParameter(DN_Parameter* p);
DN_Dim* DN_ParameterDim(DN_Parameter* p);
DN_Tensor* DN_ParameterValues(DN_Parameter* p);
DN_Tensor* DN_ParameterGradients(DN_Parameter* p);
float DN_ParameterCurrentWeightDecay(DN_Parameter* p);
void DN_SetParameterUpdated(DN_Parameter* p, bool b);
void DN_ScaleParameter(DN_Parameter* p, float s);
void DN_ScaleParameterGradient(DN_Parameter* p, float s);
bool DN_IsParameterUpdated(DN_Parameter* p);
void DN_ClipParameterInplace(DN_Parameter* p, float left, float right);
void DN_SetParameterValue(DN_Parameter* p, const float* val, int size);

// ** LookupParameter **

// **ParameterCollection**
DN_ParameterCollection* DN_NewParameterCollection();
void DN_DeleteParameterCollection(DN_ParameterCollection* pc);

DN_Parameter* DN_AddParametersToCollection(DN_ParameterCollection* pc, DN_Dim* d, const char* name);

// -----------------------------------------------------------------------------
// expr.h
typedef struct DN_Expression DN_Expression;

DN_Tensor* DN_GetExprValue(DN_Expression* e);

DN_Expression* DN_Multiply(DN_Expression* x, DN_Expression* y);
DN_Expression* DN_Logistic(DN_Expression* x);
DN_Expression* DN_BinaryLogLoss(DN_Expression* x, DN_Expression* y);

// -----------------------------------------------------------------------------
// dynet.h
// Since normally we won't modify CG directly, we don't expose ComputationGraph 
// methods here except "forward" and "backward" 
int DN_GetNumberOfActiveGraphs();
unsigned DN_GetCurrentGraphId();

typedef struct DN_ComputationGraph DN_ComputationGraph;

DN_ComputationGraph* DN_NewComputationGraph();
void DN_DeleteComputationGraph(DN_ComputationGraph* cg);

DN_Expression* DN_LoadParamToCG(DN_ComputationGraph* cg, DN_Parameter* p); //This is originaly in 'expr.h'
DN_Expression* DN_AddInputToCG(DN_ComputationGraph* cg, DN_Dim* dim, float* data, unsigned int num); //This is originaly in 'expr.h'

void DN_PrintGraphviz(DN_ComputationGraph* cg);
void DN_Backward(DN_ComputationGraph* cg, DN_Expression* last, bool full); //'full' should be false by default
float DN_Forward(DN_ComputationGraph* cg, DN_Expression* last);
unsigned DN_GetCGId(DN_ComputationGraph* cg);
void DN_SetCGCheckPoint(DN_ComputationGraph* cg);
void DN_RevertCG(DN_ComputationGraph* cg);


// -----------------------------------------------------------------------------
// training.h
typedef struct DN_SimpleSGDTrainer DN_SimpleSGDTrainer;
typedef struct DN_CyclicalSGDTrainer DN_CyclicalSGDTrainer;
typedef struct DN_MomentumSGDTrainer DN_MomentumSGDTrainer;
typedef struct DN_AdagradTrainer DN_AdagradTrainer;
typedef struct DN_AdadeltaTrainer DN_AdadeltaTrainer;
typedef struct DN_RMSPropTrainer DN_RMSPropTrainer;
typedef struct DN_AdamTrainer DN_AdamTrainer;
typedef struct DN_AmsgradTrainer DN_AmsgradTrainer;
typedef struct DN_EGTrainer DN_EGTrainer;

DN_SimpleSGDTrainer* DN_NewSimpleSGDTrainer(DN_ParameterCollection* pc, float lr);
//It should by default "lr = 0.1"
void DN_SimpleSGDUpdate(DN_SimpleSGDTrainer* trainer);


#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // DYNET_C_C_API_H_