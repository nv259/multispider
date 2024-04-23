local bert_path = 'xlm-roberta-large';

(import 'duorat-base.libsonnet')(output_from=true) {
    local PREFIX = 'dataset/spider/',
    data: {
        train_en: (import '../data/train_en.libsonnet')(prefix=PREFIX),
        train_de: (import '../data/train_de.libsonnet')(prefix=PREFIX),
        train_fr: (import '../data/train_fr.libsonnet')(prefix=PREFIX),
        train_ja: (import '../data/train_ja.libsonnet')(prefix=PREFIX),
        train_es: (import '../data/train_es.libsonnet')(prefix=PREFIX),
        train_zh: (import '../data/train_zh.libsonnet')(prefix=PREFIX),
        train_vi: (import '../data/train_vi.libsonnet')(prefix=PREFIX),
        val: (import '../data/val_en.libsonnet')(prefix=PREFIX),
        val_en: (import '../data/val_en.libsonnet')(prefix=PREFIX),
        val_de: (import '../data/val_de.libsonnet')(prefix=PREFIX),
        val_fr: (import '../data/val_fr.libsonnet')(prefix=PREFIX),
        val_ja: (import '../data/val_ja.libsonnet')(prefix=PREFIX),
        val_es: (import '../data/val_es.libsonnet')(prefix=PREFIX),
        val_zh: (import '../data/val_zh.libsonnet')(prefix=PREFIX),
        val_vi: (import '../data/val_vi.libsonnet')(prefix=PREFIX),
    },
    lr_scheduler: {
        "decay_steps": 90,
        "end_lr": 0,
        "name": "bert_warmup_polynomial",
        "num_warmup_steps": 20,
        "power": 1,
        "start_lr": 0.0001,
        "bert_factor": 8
    },
    model+: {
        name: 'InterDEMA',
        encoder: {
            num_particles: 5,
            initial_encoder: {
                name: 'Bert',
                pretrained_model_name_or_path: bert_path,
                trainable: true,
                num_return_layers: 1,
                embed_dim: 256,
                use_dedicated_gpu: false,
                use_affine_transformation: false,
                use_attention_mask: false,
                use_token_type_ids: false,
                use_position_ids: false,
                use_segments: false
            },
            "rat_attention_dropout": 0.1,
            "rat_dropout": 0.1,
            "rat_ffn_dim": 1024,
            "rat_num_heads": 8,
            "rat_num_layers": 8,
            "rat_relu_dropout": 0.1,
            source_relation_types: {
                use_schema_linking: true,
            },
            schema_input_token_ordering: '[column][table]',
            schema_source_token_ordering: '[column][table]',
            max_source_length: 200,
        },
        decoder: {
            "action_embed_dim": 64,
            "field_embed_dim": 64,
            "type_embed_dim": 64,
            "p_mask": 0.2,
            "rat_attention_dropout": 0.1,
            "rat_dropout": 0.1,
            "rat_ffn_dim": 256,
            "rat_num_heads": 8,
            "rat_num_layers": 2,
            "rat_relu_dropout": 0.1,
            pointer: {
                name: 'BahdanauMemEfficient',
                proj_size: 50,
            },
        },
        preproc+: {
            save_path: 'dataset/pkl/multilingual',
            target_vocab_pkl_path: 'dataset/pkl/multilingual/target_vocab.pkl',
            name: 'GRAPPADuoRAT',
            add_cls_token: true,
            add_sep_token: false,

            min_freq: 5,
            max_count: 5000,

            tokenizer: {
                name: 'XLMRTokenizer',
                pretrained_model_name_or_path: bert_path,
            },
            transition_system+: {
                tokenizer: {
                    name: 'XLMRTokenizer',
                    pretrained_model_name_or_path: bert_path,
                }
            },
            schema_linker+: {
                name: 'SpiderSchemaLinker',
                tokenizer: {
                   name: 'StanzaTokenizer',
                   langs: ['zh', 'ja', 'en', 'de', 'fr', 'es', 'vi'],
                },
            },
            langs: {
                train_en: 'en',
                val: 'en',
                train_zh: 'zh',
                train_ja: 'ja',
                train_de: 'de',
                train_es: 'es',
                train_fr: 'fr',
                train_vi: 'vi',
            },
        },
    },
    "train": {
        "amp_enabled": true,
        "toy_mode": false,
        "batch_size": 1,
        "n_grad_accumulation_steps": 2,
        "eval_batch_size": 16,
        "eval_beam_size": 1,
        "eval_decode_max_time_step": 500,
        "eval_every_n": 500,
        "eval_nproc": 1,
        "eval_on_train": false,
        "eval_on_val": true,
        "infer_min_n": 5000,
        "max_steps": 1000,
        "num_eval_items": 1034,
        "report_every_n": 10,
        "data_splits":  ['train_en', 'train_zh', 'train_de', 'train_fr', 'train_es', 'train_ja', 'train_vi'],
        "clip_grad": 0.3
    }
}
