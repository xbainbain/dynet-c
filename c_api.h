#ifndef DYNET_C_C_API_H_
#define DYNET_C_C_API_H_

// -----------------------------------------------------------------------------
// C API for dynet.

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

// -----------------------------------------------------------------------------
// init.h **Done!**

// By default: "shared_parameters = false"
void DN_InitializeFromArgs(int argc, char** argv, bool shared_parameters);

void DN_Cleanup();

// Resets random number generators
void DN_ResetRng(unsigned int seed);


// -----------------------------------------------------------------------------
// param-init.h **Done!**

typedef struct DN_ParameterInitNormal DN_ParameterInitNormal;
typedef struct DN_ParameterInitUniform DN_ParameterInitUniform;
typedef struct DN_ParameterInitConst DN_ParameterInitConst;
typedef struct DN_ParameterInitIdentity DN_ParameterInitIdentity;
typedef struct DN_ParameterInitGlorot DN_ParameterInitGlorot;
typedef struct DN_ParameterInitSaxe DN_ParameterInitSaxe;
typedef struct DN_ParameterInitFromFile DN_ParameterInitFromFile;
typedef struct DN_ParameterInitFromArray  DN_ParameterInitFromArray;

// Create different kind initializers for parameters
DN_ParameterInitNormal* DN_NewParameterInitNormal(float m, float v);
DN_ParameterInitUniform* DN_NewParameterInitUniform(float l, float r);
DN_ParameterInitConst* DN_NewParameterInitConst(float c);
DN_ParameterInitIdentity* DN_NewParameterInitIdentity();
DN_ParameterInitGlorot* DN_NewParameterInitGlorot(bool is_lookup, float gain);
DN_ParameterInitSaxe* DN_NewParameterInitSaxe(float gain);
DN_ParameterInitFromFile* DN_NewParameterInitFromFile(const char* f);
DN_ParameterInitFromArray* DN_NewParameterInitFromArray(const float* a,
                                                        unsigned int n);

// Destructor for initializers of parameters
void DN_DeleteParameterInitNormal(DN_ParameterInitNormal* i);
void DN_DeleteParameterInitUniform(DN_ParameterInitUniform* i);
void DN_DeleteParameterInitConst(DN_ParameterInitConst* i);
void DN_DeleteParameterInitIdentity(DN_ParameterInitIdentity* i);
void DN_DeleteParameterInitGlorot(DN_ParameterInitGlorot* i);
void DN_DeleteParameterInitSaxe(DN_ParameterInitSaxe* i);
void DN_DeleteParameterInitFromFile(DN_ParameterInitFromFile* i);
void DN_DeleteParameterInitFromArray(DN_ParameterInitFromArray* i);

// -----------------------------------------------------------------------------
// dim.h **Baisc Done**

typedef struct DN_Dim DN_Dim;

DN_Dim* DN_NewDim();

// Create a Dim object from a array of dimensions 'a' and a batch size 'b'
// 'nd' indicates the length of array
DN_Dim* DN_NewDimFromArray(const long* a, unsigned int nd, unsigned int b);

void DN_DeleteDim(DN_Dim* d);

// TODO: Other methods for Dim. Should all of them be exposed as c api?
// Need advice and discussions.

// -----------------------------------------------------------------------------
// tensor.h

typedef struct DN_Tensor DN_Tensor;

void DN_DeleteTensor(DN_Tensor* t);

DN_Dim* DN_TensorDim(DN_Tensor* t);
float* DN_TensorToArray(DN_Tensor* t);
float DN_TensorToScalar(DN_Tensor* t);
float* DN_TensorToScaleArray(DN_Tensor* t);
void DN_PrintTensor(DN_Tensor* t);

// -----------------------------------------------------------------------------
// model.h **Baisc Done**

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
void DN_ZeroLookupParameter(DN_LookupParameter* lp);
const char* DN_GetLookupParameterFullName(DN_LookupParameter* lp);
DN_Dim* DN_LookupParameterDim(DN_LookupParameter* lp);
float DN_LookupParameterCurrentWeightDecay(DN_LookupParameter* lp);
void DN_ScaleLookupParameter(DN_LookupParameter* lp, float s);
void DN_ScaleLookupParameterGradient(DN_LookupParameter* lp, float s);
void DN_SetLookupParameterUpdated(DN_LookupParameter* lp, bool b);
bool DN_IsLookupParameterUpdated(DN_LookupParameter* lp);


// **ParameterCollection**
DN_ParameterCollection* DN_NewParameterCollection();
//DN_ParameterCollection* DN_NewParameterCollectionWithWeightDecay(float weight_decay_lambda);
void DN_DeleteParameterCollection(DN_ParameterCollection* pc);

DN_Parameter* DN_AddParametersToCollectionNormal(DN_ParameterCollection* pc,
                                                 DN_Dim* d, DN_ParameterInitNormal* i,
                                                 const char* name);
DN_Parameter* DN_AddParametersToCollectionUniform(DN_ParameterCollection* pc,
                                                  DN_Dim* d, DN_ParameterInitUniform* i,
                                                  const char* name);
DN_Parameter* DN_AddParametersToCollectionConst(DN_ParameterCollection* pc,
                                                DN_Dim* d, DN_ParameterInitConst* i,
                                                const char* name);
DN_Parameter* DN_AddParametersToCollectionIdentity(DN_ParameterCollection* pc,
                                                   DN_Dim* d, DN_ParameterInitIdentity* i,
                                                   const char* name);
DN_Parameter* DN_AddParametersToCollectionGlorot(DN_ParameterCollection* pc,
                                                 DN_Dim* d, DN_ParameterInitGlorot* i,
                                                 const char* name);
DN_Parameter* DN_AddParametersToCollectionSaxe(DN_ParameterCollection* pc,
                                               DN_Dim* d, DN_ParameterInitSaxe* i,
                                               const char* name);

DN_LookupParameter* DN_AddLookupParametersToCollectionNormal(DN_ParameterCollection* pc,
                                                             unsigned int n,
                                                             DN_Dim* d,
                                                             DN_ParameterInitNormal* i,
                                                             const char* name);
DN_LookupParameter* DN_AddLookupParametersToCollectionUniform(DN_ParameterCollection* pc,
                                                              unsigned int n,
                                                              DN_Dim* d,
                                                              DN_ParameterInitUniform* i,
                                                              const char* name);
DN_LookupParameter* DN_AddLookupParametersToCollectionConst(DN_ParameterCollection* pc,
                                                            unsigned int n,
                                                            DN_Dim* d, DN_ParameterInitConst* i,
                                                            const char* name);
DN_LookupParameter* DN_AddLookupParametersToCollectionIdentity(DN_ParameterCollection* pc,
                                                               unsigned int n,
                                                               DN_Dim* d, DN_ParameterInitIdentity* i,
                                                               const char* name);
DN_LookupParameter* DN_AddLookupParametersToCollectionGlorot(DN_ParameterCollection* pc,
                                                             unsigned int n,
                                                             DN_Dim* d, DN_ParameterInitGlorot* i,
                                                             const char* name);
DN_LookupParameter* DN_AddLookupParametersToCollectionSaxe(DN_ParameterCollection* pc,
                                                           unsigned int n,
                                                           DN_Dim* d, DN_ParameterInitSaxe* i,
                                                           const char* name);

float DN_GradientL2Norm(DN_ParameterCollection* pc);
DN_ParameterCollection* DN_AddSubCollection(DN_ParameterCollection* pc,
                                            const char* name,
                                            float weight_decay_lambda);
void DN_SetWeightDecay(DN_ParameterCollection* pc, float lambda);
float DN_GetWeightDecayLambda(DN_ParameterCollection* pc);
unsigned long DN_ParameterCollectionSize(DN_ParameterCollection* pc);
const char* DN_GetParameterCollectionFullName(DN_ParameterCollection* pc);

// -----------------------------------------------------------------------------
// expr.h

typedef struct DN_Expression DN_Expression;
typedef struct DN_ComputationGraph DN_ComputationGraph; //This is originaly in 'dynet.h'

DN_Tensor* DN_GetExprValue(DN_Expression* e);
DN_Expression* DN_LoadParamToCG(DN_ComputationGraph* cg, DN_Parameter* p);
DN_Expression* DN_AddInputToCG(DN_ComputationGraph* cg, DN_Dim* dim, float* data, unsigned int num);
void DN_SetInputValueInCG(DN_Expression* e, float* data, unsigned long num);

DN_Expression* DN_Multiply(DN_Expression* x, DN_Expression* y);
DN_Expression* DN_Logistic(DN_Expression* x);
DN_Expression* DN_BinaryLogLoss(DN_Expression* x, DN_Expression* y);

// -----------------------------------------------------------------------------
// dynet.h **Basic Done!**
// Since normally we won't modify CG directly, we don't expose ComputationGraph 
// methods here except "forward" and "backward" 

int DN_GetNumberOfActiveGraphs();
unsigned DN_GetCurrentGraphId();

DN_ComputationGraph* DN_NewComputationGraph();
void DN_DeleteComputationGraph(DN_ComputationGraph* cg);

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