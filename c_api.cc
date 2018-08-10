#include "c_api_internal.h"
#include "c_api.h"


#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Macros!

#define ADD_PARAMETER(Kind) \
    DN_Parameter* DN_AddParametersToCollection##Kind(DN_ParameterCollection* pc,            \
                                                 DN_Dim* d, DN_ParameterInit##Kind* i,      \
                                                 const char* name) {                        \
    dynet::Parameter p = pc->collection.add_parameters(d->dim, i->init, std::string(name)); \
    return new DN_Parameter{p};}

#define ADD_LOOKUPPARAMETER(Kind) \
    DN_LookupParameter* DN_AddLookupParametersToCollection##Kind(DN_ParameterCollection* pc,                 \
                                                 unsigned int n,                                             \
                                                 DN_Dim* d, DN_ParameterInit##Kind* i,                       \
                                                 const char* name) {                                         \
    dynet::LookupParameter lp = pc->collection.add_lookup_parameters(n, d->dim, i->init, std::string(name)); \
    return new DN_LookupParameter{lp};}

#define DELETE(Name) \
    void DN_Delete##Name(DN_##Name* ptr) { \
        std::cout << "delete" << std::endl; \
        delete ptr; \
    }

////////////////////////////////////////////////////////////////////////////////

extern "C" {
// -----------------------------------------------------------------------------
// init.h **Done!**
void DN_InitializeFromArgs(int argc, char** argv, bool shared_parameters) {
    //It should by default "shared_parameters = false"
    dynet::initialize(argc, argv, shared_parameters);
}

void DN_Cleanup() {
    dynet::cleanup();
}

void DN_ResetRng(unsigned int seed) {
    dynet::reset_rng(seed);
}

// -----------------------------------------------------------------------------
// param-init.h **Done!**
DN_ParameterInitNormal* DN_NewParameterInitNormal(float m, float v) {
    dynet::ParameterInitNormal pi{m, v};
    return new DN_ParameterInitNormal{pi};
}

DN_ParameterInitUniform* DN_NewParameterInitUniform(float l, float r) {
    dynet::ParameterInitUniform pi{l, r};
    return new DN_ParameterInitUniform{pi};
}

DN_ParameterInitConst* DN_NewParameterInitConst(float c) {
    dynet::ParameterInitConst pi{c};
    return new DN_ParameterInitConst{pi};
}

DN_ParameterInitIdentity* DN_NewParameterInitIdentity() {
    return new DN_ParameterInitIdentity;
}

DN_ParameterInitGlorot* DN_NewParameterInitGlorot(bool is_lookup, float gain) {
    dynet::ParameterInitGlorot pi{is_lookup, gain};
    return new DN_ParameterInitGlorot{pi};
}

DN_ParameterInitSaxe* DN_NewParameterInitSaxe(float gain) {
    dynet::ParameterInitSaxe pi{gain};
    return new DN_ParameterInitSaxe{pi};
}

DN_ParameterInitFromFile* DN_NewParameterInitFromFile(const char* f) {
    dynet::ParameterInitFromFile pi{std::string(f)};
    return new DN_ParameterInitFromFile{pi};
}

DN_ParameterInitFromArray* DN_NewParameterInitFromArray(const float* a,
                                                        unsigned int n) {
    dynet::ParameterInitFromVector pi{std::vector<float>(a, a + n)};
    return new DN_ParameterInitFromArray{pi};
}

DELETE(ParameterInitNormal);
DELETE(ParameterInitUniform);
DELETE(ParameterInitConst);
DELETE(ParameterInitIdentity);
DELETE(ParameterInitGlorot);
DELETE(ParameterInitSaxe);
DELETE(ParameterInitFromFile);
DELETE(ParameterInitFromArray);

// -----------------------------------------------------------------------------
// dim.h
DN_Dim* DN_NewDim() {
    return new DN_Dim;
}

DN_Dim* DN_NewDimFromArray(const long* a, unsigned int nd, unsigned int b) {
    // The caller should make sure that the array a is not empty
    dynet::Dim dim{std::vector<long>(a, a + nd), b};
    return new DN_Dim{dim};
}

DELETE(Dim);

// TODO: Other methods for Dim

unsigned int DN_DimSize(DN_Dim* d) {
    return d->dim.size();
}

unsigned int DN_DimBatchSize(DN_Dim* d) {
    return d->dim.batch_size();
}

unsigned int DN_SumDims(DN_Dim* d) {
    return d->dim.sum_dims();
}

// -----------------------------------------------------------------------------
// tensor.h
DELETE(Tensor);

DN_Dim* DN_TensorDim(DN_Tensor* t) {
    return new DN_Dim{t->tensor.d};
}

float* DN_TensorToArray(DN_Tensor* t) {
    std::vector<float> v = dynet::as_vector(t->tensor);
    return v.data(); 
}

float DN_TensorToScalar(DN_Tensor* t) {
    return dynet::as_scalar(t->tensor);
}

/*
float* DN_TensorToScaleArray(DN_Tensor* t) {
    std::vector<float> v = dynet::as_scale_vector(t->tensor, );
    return v.data();
}*/

void DN_PrintTensor(DN_Tensor* t) {
    std::cout << t->tensor << std::endl;
}

// -----------------------------------------------------------------------------
// model.h **Baisc Done**
// ** Parameter **
DN_Parameter* DN_NewParameter() {
    return new DN_Parameter;
}

void DN_DeleteParameter(DN_Parameter* p) {
    delete p;
}

const char* DN_GetParameterFullName(DN_Parameter* p) {
    //Might be a mangle pointer! Be careful to use it!
    return p->param.get_fullname().c_str();
}

void DN_ZeroParameter(DN_Parameter* p) {
    p->param.zero();
}

DN_Dim* DN_ParameterDim(DN_Parameter* p) {   
    return new DN_Dim{p->param.dim()};
}

DN_Tensor* DN_ParameterValues(DN_Parameter* p) {
    dynet::Tensor* t = p->param.values();
    return new DN_Tensor{*t};
}

DN_Tensor* DN_ParameterGradients(DN_Parameter* p) {
    dynet::Tensor* t = p->param.gradients();
    return new DN_Tensor{*t};
}

float DN_ParameterCurrentWeightDecay(DN_Parameter* p) {
    return p->param.current_weight_decay();
}

void DN_SetParameterUpdated(DN_Parameter* p, bool b) {
    p->param.set_updated(b);
}

void DN_ScaleParameter(DN_Parameter* p, float s) {
    p->param.scale(s);
}

void DN_ScaleParameterGradient(DN_Parameter* p, float s) {
    p->param.scale_gradient(s);
}

bool DN_IsParameterUpdated(DN_Parameter* p) {
    return p->param.is_updated();
}

void DN_ClipParameterInplace(DN_Parameter* p, float left, float right) {
    p->param.clip_inplace(left, right);
}

void DN_SetParameterValue(DN_Parameter* p, const float* val, int size) {
    std::vector<float> val_vec(val, val + size);
    return p->param.set_value(val_vec);
}

// ** LookupParameter **
void DN_ZeroLookupParameter(DN_LookupParameter* lp) {
    lp->param.zero();
}

const char* DN_GetLookupParameterFullName(DN_LookupParameter* lp) {
    //Might be a mangle pointer! Be careful to use it!
    return lp->param.get_fullname().c_str();
}

DN_Dim* DN_LookupParameterDim(DN_LookupParameter* lp) {   
    return new DN_Dim{lp->param.dim()};
}

float DN_LookupParameterCurrentWeightDecay(DN_LookupParameter* lp) {
    return lp->param.current_weight_decay();
}

void DN_ScaleLookupParameter(DN_LookupParameter* lp, float s) {
    lp->param.scale(s);
}

void DN_ScaleLookupParameterGradient(DN_LookupParameter* lp, float s) {
    lp->param.scale_gradient(s);
}

void DN_SetLookupParameterUpdated(DN_LookupParameter* lp, bool b) {
    lp->param.set_updated(b);
}

bool DN_IsLookupParameterUpdated(DN_LookupParameter* lp) {
    return lp->param.is_updated();
}

// ** ParameterCollection **
DN_ParameterCollection* DN_NewParameterCollection() {
    return new DN_ParameterCollection;
}

/* There is a known bug in the c++ source file
DN_ParameterCollection* DN_NewParameterCollectionWithWeightDecay(float weight_decay_lambda) {
    dynet::ParameterCollection pc{weight_decay_lambda};
    return new DN_ParameterCollection{pc};
}
*/

void DN_DeleteParameterCollection(DN_ParameterCollection* pc) {
    delete pc;
}

ADD_PARAMETER(Normal);
ADD_PARAMETER(Uniform);
ADD_PARAMETER(Const);
ADD_PARAMETER(Identity);
ADD_PARAMETER(Glorot);
ADD_PARAMETER(Saxe);

ADD_LOOKUPPARAMETER(Normal);
ADD_LOOKUPPARAMETER(Uniform);
ADD_LOOKUPPARAMETER(Const);
ADD_LOOKUPPARAMETER(Identity);
ADD_LOOKUPPARAMETER(Glorot);
ADD_LOOKUPPARAMETER(Saxe);

DN_ParameterCollection* DN_AddSubCollection(DN_ParameterCollection* pc,
                                            const char* name,
                                            float weight_decay_lambda) {
    dynet::ParameterCollection _pc = pc->collection.add_subcollection(
        std::string(name), weight_decay_lambda);
    return new DN_ParameterCollection{_pc};
}

float DN_GradientL2Norm(DN_ParameterCollection* pc) {
    return pc->collection.gradient_l2_norm();
}

void DN_SetWeightDecay(DN_ParameterCollection* pc, float lambda) {
    pc->collection.set_weight_decay_lambda(lambda);
}

float DN_GetWeightDecayLambda(DN_ParameterCollection* pc) {
    return pc->collection.get_weight_decay_lambda();
}


unsigned long DN_ParameterCollectionSize(DN_ParameterCollection* pc) {
    return pc->collection.size();
}

const char* DN_GetParameterCollectionFullName(DN_ParameterCollection* pc) {
    //Might be a mangle pointer! Be careful to use it!
    return pc->collection.get_fullname().c_str();
}


// -----------------------------------------------------------------------------
// dynet.h **Basic Done!**
int DN_GetNumberOfActiveGraphs() {
    return dynet::get_number_of_active_graphs();
}

unsigned DN_GetCurrentGraphId() {
    return dynet::get_current_graph_id();
}

DN_ComputationGraph* DN_NewComputationGraph() {
    return new DN_ComputationGraph;
}

void DN_DeleteComputationGraph(DN_ComputationGraph* cg) {
    delete cg;
}

void DN_PrintGraphviz(DN_ComputationGraph* cg) {
    cg->graph.print_graphviz();
}

void DN_Backward(DN_ComputationGraph* cg, DN_Expression* last, bool full) {
    cg->graph.backward(last->expr, full);
}

float DN_Forward(DN_ComputationGraph* cg, DN_Expression* last) {
    return dynet::as_scalar(cg->graph.forward(last->expr));
}

unsigned DN_GetCGId(DN_ComputationGraph* cg) {
    return cg->graph.get_id();
}

void DN_SetCGCheckPoint(DN_ComputationGraph* cg) {
    cg->graph.checkpoint();
}

void DN_RevertCG(DN_ComputationGraph* cg) {
    cg->graph.revert();
}

// -----------------------------------------------------------------------------
// expr.h
DN_Tensor* DN_GetExprValue(DN_Expression* e) {
    return new DN_Tensor{e->expr.value()};
}

DN_Expression* DN_LoadParamToCG(DN_ComputationGraph* cg, DN_Parameter* p) {
    dynet::Expression e = dynet::parameter(cg->graph, p->param);
    return new DN_Expression{e};
}


DN_Expression* DN_AddInputToCG(DN_ComputationGraph* cg,
                               DN_Dim* dim,
                               float* data,
                               unsigned int num) {
    // This implementation has a known issue:change outside array can not change
    // the actual internal value since the new vector copy the data in array!
    // Try to create a new function like 'set' in Python!
    
    // Transforme array to vector.
    std::vector<float> data_v(data, data + num);
    dynet::Expression e = dynet::input(cg->graph, dim->dim, data_v);
    return new DN_Expression{e};
}

void DN_SetInputValueInCG(DN_Expression* e, float* vals, unsigned long num) {
    // Don't know if it is the correct way to do that!!!
    // Have a known bug!!! Don't know how to fix it now!
    std::vector<float> vals_vec(vals, vals + num);
    //std::cout << e->expr.pg->get_value(e->expr) << std::endl;
    dynet::TensorTools::set_elements(e->expr.pg->get_value(e->expr), vals_vec);
    //std::cout << e->expr.pg->get_value(e->expr) << std::endl;
}


DN_Expression* DN_Multiply(DN_Expression* x, DN_Expression* y) {
    dynet::Expression e = x->expr * y->expr;
    return new DN_Expression{e};
}

DN_Expression* DN_Logistic(DN_Expression* x) {
    dynet::Expression e = dynet::logistic(x->expr);
    return new DN_Expression{e};
}

DN_Expression* DN_BinaryLogLoss(DN_Expression* x, DN_Expression* y) {
    dynet::Expression e = dynet::binary_log_loss(x->expr, y->expr);
    return new DN_Expression{e};
}



// -----------------------------------------------------------------------------
// training.h

DN_SimpleSGDTrainer* DN_NewSimpleSGDTrainer(DN_ParameterCollection* pc,
                                            float lr) {
    dynet::SimpleSGDTrainer trainer(pc->collection, lr);
    return new DN_SimpleSGDTrainer{trainer};
}

DN_CyclicalSGDTrainer* DN_NewCyclicalSGDTrainer(DN_ParameterCollection* pc,
                                                float lr_min, float lr_max,
                                                float step_size,
                                                float gamma, float  edecay) {
    dynet::CyclicalSGDTrainer trainer(pc->collection, lr_min, lr_max, 
        step_size, gamma, edecay);
    return new DN_CyclicalSGDTrainer{trainer};
}

DN_MomentumSGDTrainer* DN_NewMomentumSGDTrainer(DN_ParameterCollection* pc,
                                                float lr, float mom) {
    dynet::MomentumSGDTrainer trainer(pc->collection, lr, mom);
    return new DN_MomentumSGDTrainer{trainer};
}

DN_AdagradTrainer* DN_NewAdagradTrainer(DN_ParameterCollection* pc,
                                        float lr, float eps) {
    dynet::AdagradTrainer trainer(pc->collection, lr, eps);
    return new DN_AdagradTrainer{trainer};
}

DN_AdadeltaTrainer* DN_NewAdadeltaTrainer(DN_ParameterCollection* pc,
                                          float eps, float rho) {
    dynet::AdadeltaTrainer trainer(pc->collection, eps, rho);
    return new DN_AdadeltaTrainer{trainer};
}

DN_RMSPropTrainer* DN_NewRMSPropTrainer(DN_ParameterCollection* pc, float lr,
                                        float eps, float rho) {
    dynet::RMSPropTrainer trainer(pc->collection, lr, eps, rho);
    return new DN_RMSPropTrainer{trainer};
}

DN_AdamTrainer* DN_NewAdamTrainer(DN_ParameterCollection* pc, float lr,
                                  float beta_1, float beta_2, float eps) {
    dynet::AdamTrainer trainer(pc->collection, lr, beta_1, beta_2, eps);
    return new DN_AdamTrainer{trainer};
}

DN_AmsgradTrainer* DN_NewAmsgradTrainer(DN_ParameterCollection* pc, float lr,
                                        float beta_1, float beta_2, float eps) {
    dynet::AmsgradTrainer trainer(pc->collection, lr, beta_1, beta_2, eps);
    return new DN_AmsgradTrainer{trainer};
}

DELETE(SimpleSGDTrainer);
DELETE(CyclicalSGDTrainer);
DELETE(MomentumSGDTrainer);
DELETE(AdagradTrainer);
DELETE(AdadeltaTrainer);
DELETE(RMSPropTrainer);
DELETE(AdamTrainer);
DELETE(AmsgradTrainer);

// Use marcro if more methods need to be added
void DN_SimpleSGDTrainerUpdate(DN_SimpleSGDTrainer* trainer) {
    trainer->trainer.update();
}
void DN_CyclicalSGDTrainerUpdate(DN_CyclicalSGDTrainer* trainer) {
    trainer->trainer.update();
}
void DN_MomentumSGDTrainerUpdate(DN_MomentumSGDTrainer* trainer) {
    trainer->trainer.update();
}
void DN_AdagradTrainerUpdate(DN_AdagradTrainer* trainer) {
    trainer->trainer.update();
}
void DN_RMSPropTrainerUpdate(DN_RMSPropTrainer* trainer) {
    trainer->trainer.update();
}
void DN_AdamTrainerUpdate(DN_AdamTrainer* trainer) {
    trainer->trainer.update();
}
void DN_AmsgradTrainerUpdate(DN_AmsgradTrainer* trainer) {
    trainer->trainer.update();
}

float DN_SimpleSGDTrainerClipGradients(DN_SimpleSGDTrainer* trainer) {
    return trainer->trainer.clip_gradients();
}
float DN_CyclicalSGDTrainerClipGradients(DN_CyclicalSGDTrainer* trainer) {
    return trainer->trainer.clip_gradients();
}
float DN_MomentumSGDTrainerClipGradients(DN_MomentumSGDTrainer* trainer) {
    return trainer->trainer.clip_gradients();
}
float DN_AdagradTrainerClipGradients(DN_AdagradTrainer* trainer) {
    return trainer->trainer.clip_gradients();
}
float DN_RMSPropTrainerClipGradients(DN_RMSPropTrainer* trainer) {
    return trainer->trainer.clip_gradients();
}
float DN_AdamTrainerClipGradients(DN_AdamTrainer* trainer) {
    return trainer->trainer.clip_gradients();
}
float DN_AmsgradTrainerClipGradients(DN_AmsgradTrainer* trainer) {
    return trainer->trainer.clip_gradients();
}


// 

} // end extern "C"