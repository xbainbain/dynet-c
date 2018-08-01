#ifndef DYNETC_C_API_INTERNAL_H_
#define DYNETC_C_API_INTERNAL_H_


#include "dynet/init.h"
#include "dynet/param-init.h"
#include "dynet/tensor.h"
#include "dynet/dim.h"
#include "dynet/model.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/training.h"
#include "dynet/graph.h"

// -----------------------------------------------------------------------------
// Internal structures used by the C API.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// init.h
struct DN_DynetParams {
    dynet::DynetParams params;
};

// -----------------------------------------------------------------------------
// param-init.h
struct DN_ParameterInitNormal {
    dynet::ParameterInitNormal init;
};

struct DN_ParameterInitUniform {
    dynet::ParameterInitUniform init;
};

struct DN_ParameterInitConst {
    dynet::ParameterInitConst init;
};

struct DN_ParameterInitIdentity {
    dynet::ParameterInitIdentity init;
};

struct DN_ParameterInitGlorot {
    dynet::ParameterInitGlorot init;
};

struct DN_ParameterInitSaxe {
    dynet::ParameterInitSaxe init;
};

struct DN_ParameterInitFromFile {
    dynet::ParameterInitFromFile init;
};

struct DN_ParameterInitFromArray {
    dynet::ParameterInitFromVector init;
};

// -----------------------------------------------------------------------------
// tensor.h
struct DN_Tensor {
    dynet::Tensor tensor;
};

// -----------------------------------------------------------------------------
// dim.h
struct DN_Dim {
    dynet::Dim dim;
};

// -----------------------------------------------------------------------------
// model.h
struct DN_Parameter {
    dynet::Parameter param;
};

struct DN_LookupParameter {
    dynet::LookupParameter param;
};

struct DN_ParameterCollection {
    //DN_ParameterCollection();
   // DN_ParameterCollection(float weight_decay_lambda);
    
    dynet::ParameterCollection collection;
};

// -----------------------------------------------------------------------------
// dynet.h
struct DN_ComputationGraph {
    dynet::ComputationGraph graph;
};

// -----------------------------------------------------------------------------
// expr.h
struct DN_Expression {
    dynet::Expression expr;
};

// -----------------------------------------------------------------------------
// training.h
struct DN_SimpleSGDTrainer {
    DN_SimpleSGDTrainer(DN_ParameterCollection* pc, float learning_rate)
        : trainer(pc->collection, learning_rate){}
    dynet::SimpleSGDTrainer trainer;
};

/*
struct DN_CyclicalSGDTrainer {
    DN_CyclicalSGDTrainer(DN_ParameterCollection* pc, float learning_rate)
        : trainer(pc->collection, learning_rate){}
    dynet::CyclicalSGDTrainer trainer;
};

struct DN_MomentumSGDTrainer {
    DN_MomentumSGDTrainer(DN_ParameterCollection* pc, float learning_rate)
        : trainer(pc->collection, learning_rate){}
    dynet::MomentumSGDTrainer trainer;
};

struct DN_AdagradTrainer {
    DN_AdagradTrainer(DN_ParameterCollection* pc, float learning_rate)
        : trainer(pc->collection, learning_rate){}
    dynet::AdagradTrainer trainer;
};

struct DN_AdadeltaTrainer {
    DN_AdadeltaTrainer(DN_ParameterCollection* pc, float learning_rate)
        : trainer(pc->collection, learning_rate){}
    dynet::AdadeltaTrainer trainer;
};

struct DN_RMSPropTrainer {
    DN_RMSPropTrainer(DN_ParameterCollection* pc, float learning_rate)
        : trainer(pc->collection, learning_rate){}
    dynet::RMSPropTrainer trainer;
};

struct DN_AdamTrainer {
    DN_AdamTrainer(DN_ParameterCollection* pc, float learning_rate)
        : trainer(pc->collection, learning_rate){}
    dynet::AdamTrainer trainer;
};

struct DN_AmsgradTrainer {
    DN_AmsgradTrainer(DN_ParameterCollection* pc, float learning_rate)
        : trainer(pc->collection, learning_rate){}
    dynet::AmsgradTrainer trainer;
};

struct DN_EGTrainer {
    DN_EGTrainer(DN_ParameterCollection* pc, float learning_rate)
        : trainer(pc->collection, learning_rate){}
    dynet::EGTrainer trainer;
};
*/

#endif  // DYNETC_C_API_INTERNAL_H_