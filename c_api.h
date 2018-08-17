#ifndef DYNET_C_C_API_H_
#define DYNET_C_C_API_H_

// -----------------------------------------------------------------------------
// C API for dynet.

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdlib.h>

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
size_t DN_ParameterCollectionSize(DN_ParameterCollection* pc);
const char* DN_GetParameterCollectionFullName(DN_ParameterCollection* pc);

// -----------------------------------------------------------------------------
// expr.h

typedef struct DN_Expression DN_Expression;
typedef struct DN_ComputationGraph DN_ComputationGraph; //This is originaly in 'dynet.h'

void DN_DeleteExpression(DN_Expression* e);
DN_Tensor* DN_GetExprValue(DN_Expression* e);
DN_Expression* DN_LoadParamToCG(DN_ComputationGraph* cg, DN_Parameter* p);
DN_Expression* DN_AddInputToCG(DN_ComputationGraph* cg, DN_Dim* dim, const float* data, size_t num);
void DN_SetInputValueInCG(DN_Expression* e, float* data, size_t num);

DN_Expression* DN_Multiply(DN_Expression* x, DN_Expression* y);
DN_Expression* DN_Logistic(DN_Expression* x);
DN_Expression* DN_BinaryLogLoss(DN_Expression* x, DN_Expression* y);

////////////////////////////////////////////////
// Arithmetic operations                      //
////////////////////////////////////////////////

DN_Expression* DN_Negate(DN_Expression* x);
DN_Expression* DN_Add(DN_Expression* x, DN_Expression* y);
DN_Expression* DN_AddWithScalar(DN_Expression* x, float y);
DN_Expression* DN_Subtract(DN_Expression* x, DN_Expression* y);
DN_Expression* DN_SubtractFromScalar(float x, DN_Expression* y);
DN_Expression* DN_SubtractByScalar(DN_Expression* x, float y);
DN_Expression* DN_Multiply(DN_Expression* x, DN_Expression* y);
DN_Expression* DN_MultiplyWithScalar(DN_Expression* x, float y);
DN_Expression* DN_Divide(DN_Expression* x, DN_Expression* y);
DN_Expression* DN_DivideByScalar(DN_Expression* x, float y);
DN_Expression* DN_AffineTransform(DN_Expression* xs[], int num);
DN_Expression* DN_Sum(DN_Expression* xs[], int num);
DN_Expression* DN_SumElems(DN_Expression* x);
DN_Expression* DN_MomentElems(DN_Expression* x, unsigned int r);
DN_Expression* DN_MomentBatches(DN_Expression* x, unsigned int r);
DN_Expression* DN_MeanElems(DN_Expression* x);
DN_Expression* DN_StdElems(DN_Expression* x);
DN_Expression* DN_SumBatches(DN_Expression* x);
DN_Expression* DN_MeanBatches(DN_Expression* x);
DN_Expression* DN_StdBatches(DN_Expression* x);
DN_Expression* DN_Cumsum(DN_Expression* x, unsigned int d);
//............

DN_Expression* DN_Average(DN_Expression* xs[], int num);
DN_Expression* DN_Sqrt(DN_Expression* x);
DN_Expression* DN_Abs(DN_Expression* x);
DN_Expression* DN_Erf(DN_Expression* x);
DN_Expression* DN_Asin(DN_Expression* x);
DN_Expression* DN_Acos(DN_Expression* x);
DN_Expression* DN_Atan(DN_Expression* x);
DN_Expression* DN_Sin(DN_Expression* x);
DN_Expression* DN_Cos(DN_Expression* x);
DN_Expression* DN_Tan(DN_Expression* x);
DN_Expression* DN_Sinh(DN_Expression* x);
DN_Expression* DN_Cosh(DN_Expression* x);
DN_Expression* DN_Tanh(DN_Expression* x);
DN_Expression* DN_Asinh(DN_Expression* x);
DN_Expression* DN_Acosh(DN_Expression* x);
DN_Expression* DN_Exp(DN_Expression* x);
DN_Expression* DN_Square(DN_Expression* x);
DN_Expression* DN_Cube(DN_Expression* x);
DN_Expression* DN_LogSigmoid(DN_Expression* x);
DN_Expression* DN_Lgamma(DN_Expression* x);
DN_Expression* DN_Log(DN_Expression* x);
DN_Expression* DN_Logistic(DN_Expression* x);
DN_Expression* DN_Rectify(DN_Expression* x);
DN_Expression* DN_Selu(DN_Expression* x);
DN_Expression* DN_Elu(DN_Expression* x, float alpha);
DN_Expression* DN_Silu(DN_Expression* x, float beta);
DN_Expression* DN_Softsign(DN_Expression* x);
DN_Expression* DN_Pow(DN_Expression* x, DN_Expression* y);
DN_Expression* DN_Min(DN_Expression* x, DN_Expression* y);
DN_Expression* DN_Max(DN_Expression* x, DN_Expression* y);
DN_Expression* DN_MaxArray(DN_Expression* xs[], int num);
DN_Expression* DN_DotProduct(DN_Expression* x, DN_Expression* y);
DN_Expression* DN_CircConv(DN_Expression* x, DN_Expression* y);
DN_Expression* DN_CircCorr(DN_Expression* x, DN_Expression* y);
DN_Expression* DN_Cmult(DN_Expression* x, DN_Expression* y);
DN_Expression* DN_Cdiv(DN_Expression* x, DN_Expression* y);
DN_Expression* DN_ColwiseAdd(DN_Expression* x, DN_Expression* y);

////////////////////////////////////////////////
// Probability/loss operations                //
////////////////////////////////////////////////

DN_Expression* DN_Softmax(DN_Expression* x, unsigned int d);
DN_Expression* DN_LogsumexpDim(DN_Expression* x, unsigned int d);
DN_Expression* DN_LogSoftmax(DN_Expression* x);
DN_Expression* DN_Logsumexp(DN_Expression* xs[], int num);
DN_Expression* DN_BinaryLogLoss(DN_Expression* x, DN_Expression* y);
DN_Expression* DN_SquaredDistance(DN_Expression* x, DN_Expression* y);

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
// training.h **Done**

typedef struct DN_SimpleSGDTrainer DN_SimpleSGDTrainer;
typedef struct DN_CyclicalSGDTrainer DN_CyclicalSGDTrainer;
typedef struct DN_MomentumSGDTrainer DN_MomentumSGDTrainer;
typedef struct DN_AdagradTrainer DN_AdagradTrainer;
typedef struct DN_AdadeltaTrainer DN_AdadeltaTrainer;
typedef struct DN_RMSPropTrainer DN_RMSPropTrainer;
typedef struct DN_AdamTrainer DN_AdamTrainer;
typedef struct DN_AmsgradTrainer DN_AmsgradTrainer;

// Create new Stochastic gradient descent trainer.
// Default values: lr=0.1
DN_SimpleSGDTrainer* DN_NewSimpleSGDTrainer(DN_ParameterCollection* pc,
                                            float lr);

// Cyclical learning rate SGD.
// Default values: lr_min=0.01, lr_max=0.1, step_size=2000, gamma=1.0, edecay=0.0
DN_CyclicalSGDTrainer* DN_NewCyclicalSGDTrainer(DN_ParameterCollection* pc,
                                                float lr_min, float lr_max,
                                                float step_size,
                                                float gamma, float  edecay);

// Stochastic gradient descent with momentum.
// Default values: lr=0.01, mom=0.9
DN_MomentumSGDTrainer* DN_NewMomentumSGDTrainer(DN_ParameterCollection* pc,
                                                float lr, float mom);

// Adagrad optimizer.
// Default values: lr=0.1, eps=1e-20
DN_AdagradTrainer* DN_NewAdagradTrainer(DN_ParameterCollection* pc,
                                        float lr, float eps);

// AdaDelta optimizer.
// Default values: eps=1e-6, rho=0.95
DN_AdadeltaTrainer* DN_NewAdadeltaTrainer(DN_ParameterCollection* pc,
                                          float eps, float rho);

// RMSProp optimizer.
// Default values: lr=0.1, eps=1e-20, rho=0.95
DN_RMSPropTrainer* DN_NewRMSPropTrainer(DN_ParameterCollection* pc, float lr,
                                        float eps, float rho);

// Adam optimizer.
// Default values: lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8
DN_AdamTrainer* DN_NewAdamTrainer(DN_ParameterCollection* pc, float lr,
                                  float beta_1, float beta_2, float eps);

// AMSGrad optimizer.
// Default values: lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8
DN_AmsgradTrainer* DN_NewAmsgradTrainer(DN_ParameterCollection* pc, float lr,
                                        float beta_1, float beta_2, float eps);


// Destructor for trainers
void DN_DeleteSimpleSGDTrainer(DN_SimpleSGDTrainer* t);
void DN_DeleteCyclicalSGDTrainer(DN_CyclicalSGDTrainer* t);
void DN_DeleteMomentumSGDTrainer(DN_MomentumSGDTrainer* t);
void DN_DeleteAdagradTrainer(DN_AdagradTrainer* t);
void DN_DeleteAdadeltaTrainer(DN_AdadeltaTrainer* t);
void DN_DeleteRMSPropTrainer(DN_RMSPropTrainer* t);
void DN_DeleteAdamTrainer(DN_AdamTrainer* t);
void DN_DeleteAmsgradTrainer(DN_AmsgradTrainer* t);

// Update the parameters according to the appropriate update rule
void DN_SimpleSGDTrainerUpdate(DN_SimpleSGDTrainer* trainer);
void DN_CyclicalSGDTrainerUpdate(DN_CyclicalSGDTrainer* trainer);
void DN_MomentumSGDTrainerUpdate(DN_MomentumSGDTrainer* trainer);
void DN_AdagradTrainerUpdate(DN_AdagradTrainer* trainer);
void DN_RMSPropTrainerUpdate(DN_RMSPropTrainer* trainer);
void DN_AdamTrainerUpdate(DN_AdamTrainer* trainer);
void DN_AmsgradTrainerUpdate(DN_AmsgradTrainer* trainer);

// Clip gradient
float DN_SimpleSGDTrainerClipGradients(DN_SimpleSGDTrainer* trainer);
float DN_CyclicalSGDTrainerClipGradients(DN_CyclicalSGDTrainer* trainer);
float DN_MomentumSGDTrainerClipGradients(DN_MomentumSGDTrainer* trainer);
float DN_AdagradTrainerClipGradients(DN_AdagradTrainer* trainer);
float DN_RMSPropTrainerClipGradients(DN_RMSPropTrainer* trainer);
float DN_AdamTrainerClipGradients(DN_AdamTrainer* trainer);
float DN_AmsgradTrainerClipGradients(DN_AmsgradTrainer* trainer);


#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // DYNET_C_C_API_H_