#include "c_api.h"
#include "c_api_internal.h"

#include <vector>

extern "C" {
// -----------------------------------------------------------------------------
// init.h **Done!**
DN_DynetParams* DN_NewDynetParams() {
    return new DN_DynetParams;
}

void DN_DeleteDynetParams(DN_DynetParams* dp) {
    delete dp;
}

void DN_InitializeFromParams(DN_DynetParams* dp) {
    dynet::initialize(dp->params);
}

void DN_InitializeFromArgs(int argc, char** argv, bool shared_parameters) {
    //It should by default "shared_parameters = false"
    dynet::initialize(argc, argv, shared_parameters);
}

void DN_Cleanup() {
    dynet::cleanup();
}

void DN_ResetRng(unsigned seed) {
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
                                                        size_t n) {
    dynet::ParameterInitFromVector pi{std::vector<float>(a, a + n)};
    return new DN_ParameterInitFromArray{pi};
}

// -----------------------------------------------------------------------------
// dim.h
DN_Dim* DN_NewDim() {
    return new DN_Dim;
}

DN_Dim* DN_NewDimFromArray(const long a[], size_t n, unsigned int b) {
    // The caller should make sure that the array a is not empty
    dynet::Dim dim{std::vector<long>(a, a + n), b};
    return new DN_Dim{dim};
}

void DN_DeleteDim(DN_Dim* dim) {
    delete dim;
}




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
void DN_DeleteTensor(DN_Tensor* t) {
    delete t;
}

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
// model.h
// ** Parameter **
DN_Parameter* DN_NewParameter() {
    return new DN_Parameter;
}

void DN_DeleteParameter(DN_Parameter* p) {
    delete p;
}

const char* DN_GetParameterFullName(DN_Parameter* p) {
    std::string name = p->param.get_fullname();
    return name.c_str();
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


// ** ParameterCollection **
DN_ParameterCollection* DN_NewParameterCollection() {
    return new DN_ParameterCollection;
}

void DN_DeleteParameterCollection(DN_ParameterCollection* pc) {
    delete pc;
}

DN_Parameter* DN_AddParametersToCollection(DN_ParameterCollection* pc, DN_Dim* d, char* name) {
    dynet::Parameter p = pc->collection.add_parameters(d->dim, std::string(name));
    return new DN_Parameter{p};
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


DN_Expression* DN_AddInputToCG(DN_ComputationGraph* cg, DN_Dim* dim, float* data, unsigned int num) {
    // This implementation has a known bug:change outside array can not change
    // the actual internal value since the new vector copy the data in array!
    // The other way to do it is wrap c++ std::vector into c.(Will do it later)
    
    // Transforme array to vector.
    std::vector<float> data_v(data, data + num);
    dynet::Expression e = dynet::input(cg->graph, dim->dim, data_v);
    return new DN_Expression{e};
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
DN_SimpleSGDTrainer* DN_NewSimpleSGDTrainer(DN_ParameterCollection* pc, float lr) {
    return new DN_SimpleSGDTrainer(pc, lr);
}

void DN_SimpleSGDUpdate(DN_SimpleSGDTrainer* trainer) {
    trainer->trainer.update();
}

} // end extern "C"