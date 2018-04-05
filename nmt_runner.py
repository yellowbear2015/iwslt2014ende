#!/usr/bin/env python
# -*- coding:utf8 -*-

# ================================================================================
# Copyright 2016 Alibaba Inc. All Rights Reserved.
#
# @self: Train a NMT model or decoding test data using trained models.
#
# Version: 0.1
# History:
#   Created by:
#       mark.zhangh(email:mark.zhangh@alibaba-inc.com), 2017-01-11 09:38
#   Last modified:
# ================================================================================

from config import get_config
from data_reader import DataReader
from nmt_model import NmtModel
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.session_bundle import exporter

import os, sys
import subprocess
import time
import numpy

''' NMT system runner.
'''
class NmtSystemRunner(object):
# private:

    ''' Init function.
    '''
    def __init__(self):
        ''' system configurations '''
        self._config = get_config()
        ''' system action type '''
        self._action_type = self._config.action_type
        if self._action_type not in self._config.action_type_list.split():
            raise AttributeError("%s: Unsupported argument 'action_type' value '%s'" %
                                    (type(self).__name__, self.action_type))
        ''' training arguments '''
        self._max_epochs = self._config.max_epochs
        self._steps_per_ckpt = self._config.steps_per_ckpt
        self._steps_per_validation = self._config.steps_per_validation
        self._steps_per_sampling = self._config.steps_per_sampling
        self._sampling_num = self._config.sampling_num
        ''' decoding arguments '''
        self._beam_size = self._config.beam_size
        self._max_decoded_trg_len = self._config.max_decoded_trg_len
        self._normalize_score_flag = self._config.normalize_score
        ''' create a tensorflow session config '''
        self._tf_config = tf.ConfigProto()
        self._tf_config.log_device_placement = self._config.log_device_placement
        self._tf_config.allow_soft_placement = self._config.allow_soft_placement
        self._tf_config.gpu_options.allow_growth = self._config.allow_gpu_growth
        if tf.gfile.Exists(self._config.tf_log_dir):
            tf.gfile.DeleteRecursively(self._config.tf_log_dir)
        tf.gfile.MakeDirs(self._config.tf_log_dir)

# public:

    def run(self):
        with tf.device('/gpu:6'):
            if 'trainer' == self._action_type:
                self._train()
            elif 'force_decoder' == self._action_type:
                self._force_decode()
            elif 'decoder' == self._action_type:
                if 'export_single' == self._config.export_type:
                    self._export_single()
                else:
                    self._decode()
                '''
                self._decode()
                '''
            else:
                pass

    def _train(self):
        with tf.Session(config=self._tf_config) as tf_session:
            logging.info("%s: Begin NMT training at time: %s ..." \
                            % (type(self).__name__, time.asctime(time.localtime(time.time()))))
            # create a nmt model
            nmt_model = NmtModel(tf_session, self._config)
            # create a data reader and get vocab ID to token dictionaries
            data_reader = DataReader(self._config)
            src_id2vocab_dict = data_reader.get_id2vocab_dict(target_side=False)
            trg_id2vocab_dict = data_reader.get_id2vocab_dict()
            save_idx = 0
            save_num = 10
            for _epoch in range(self._max_epochs):
                _step = 0
                for (src, src_mask), (trg, trg_mask) in data_reader.get_next_data_batch('train'):
                    # train model parameters by a data batch
                    cur_time = time.time()
                    train_cost, model_cost, updates = nmt_model.batch_training(tf_session, \
                        src, src_mask, trg, trg_mask, False)
                    elapsed_time = time.time() - cur_time
                    step_speed = elapsed_time / float(len(src))
                    logging.info("%s: epoch: %d, step: %d, train cost: %.4f, model cost: %.4f, time: %.4f sec, speed: %.4f sec/sent." % (type(self).__name__, _epoch, _step, train_cost, model_cost, elapsed_time, step_speed))
                    if 0 == (_step + 1) % self._steps_per_ckpt:
                        # save model checkpoint
                        logging.info("%s: Saving model on epoch: %d, step: %d." \
                            % (type(self).__name__, _epoch, _step))
                        save_idx += 1
                        model_suffix = str(save_idx % save_num)
                        nmt_model.save_model(tf_session, model_suffix, False)
                    if 0 == (_step + 1) % self._steps_per_sampling:
                        # sampling
                        logging.info("%s: Sampling on epoch: %d, step: %d." \
                            % (type(self).__name__, _epoch, _step))
                        self.__sampling(tf_session, nmt_model, self._sampling_num, \
                            src_id2vocab_dict, trg_id2vocab_dict, src, src_mask, \
                            self._beam_size, self._max_decoded_trg_len, trg, trg_mask)
                    if 0 == (_step + 1) % self._steps_per_validation:
                        # validation
                        logging.info("%s: Validating on epoch: %d, step: %d." \
                            % (type(self).__name__, _epoch, _step))
                        tag='.trans.epoch-%d.step-%d' % (_epoch, _step)
                        valid_out_file=data_reader.src_test_file+tag
                        with open(valid_out_file, 'w') as fout:
                            self._decode(tf_session, nmt_model, data_reader, None, fout)
                        bleu_score = self._evaluate_bleu(data_reader.src_test_file, \
                            data_reader.trg_test_file, valid_out_file)
                        logging.info("%s: BLEU score: %s" % (type(self).__name__, bleu_score))
                    _step += 1
            nmt_model.save_model(tf_session, '', True)
            logging.info("%s: NMT training completed at time: %s." \
                            % (type(self).__name__, time.asctime(time.localtime(time.time()))))

    def _evaluate_bleu(self, src, ref, tst):
        script_file=os.path.join(os.path.dirname(__file__), 'tools/evaluate_bleu.sh')
        if 1 == self._config.test_file_num:
            cmd = 'bash %s %s %s %s' % (script_file, src, ref, tst)
            try:
                p = subprocess.Popen(cmd, shell=True, universal_newlines=True, \
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                out, err = p.communicate()
                assert p.returncode == 0, 'Evaluation Script Failed: %s : out=%s err=%s' \
                    % (cmd, out, err)
                return "%0.4f"%(float(out.split()[0]))
            except Exception as e:
                logging.info("%s: Got Exception: %s" % (type(self).__name__, e))
                return str(-1.0)
        else:
            bleu_score = ""
            tstoutlines = open(tst).readlines()
            cur_total_num = 0
            for i in range(self._config.test_file_num):
                cur_num = len(open(src+'.'+str(i), 'rb').readlines())
                outf = open(tst+'.'+str(i), 'w')
                for line in tstoutlines[cur_total_num:cur_total_num+cur_num]:
                    outf.write(line.strip()+'\n')
                outf.close()
                cur_total_num += cur_num
                cmd = 'bash %s %s %s %s' % (script_file, src+'.'+str(i), ref+'.'+str(i), tst+'.'+str(i))
                try:
                    p = subprocess.Popen(cmd, shell=True, universal_newlines=True, \
                        stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    out, err = p.communicate()
                    assert p.returncode == 0, 'Evaluation Script Failed: %s : out=%s err=%s' \
                        % (cmd, out, err)
                    bleu_score +=  "%0.4f\t" % (float(out.split()[0]))
                except Exception as e:
                    logging.info("%s: Got Exception: %s" % (type(self).__name__, e))
                    bleu_score += str(-1.0) + '\t'
            return bleu_score

    def _force_decode(self, fout=sys.stdout):
        with tf.Session(config=self._tf_config) as tf_session:
            logging.info("%s: Begin NMT force-decoding at time: %s ..." \
                            % (type(self).__name__, time.asctime(time.localtime(time.time()))))
            # create or restore a nmt model
            nmt_model = NmtModel(tf_session, self._config)
            # create a data reader
            data_reader = DataReader(self._config)
            cur_time = time.time()
            _step = 0
            line_num = 0
            for (src, src_mask), (trg, trg_mask) in data_reader.get_next_data_batch('test'):
                # batch size is len(src)
                line_num += len(src)
                predict_cost = nmt_model.batch_training(tf_session, \
                    src, src_mask, trg, trg_mask, True)
                # dump prediction results
                print >> fout, "cost: %s\n" % (predict_cost)
                if 0 == (_step + 1) % self._steps_per_ckpt:
                    elapsed_time = time.time() - cur_time
                    decode_speed = elapsed_time / float(line_num)
                    logging.info("%s: %d sentence-pairs force decoded, elapsed time: %.4f sec, speed: %.4f sec/sent." % (type(self).__name__, line_num, elapsed_time, decode_speed))
                _step += 1
            logging.info("%s: NMT force-decoding completed at time: %s." \
                            % (type(self).__name__, time.asctime(time.localtime(time.time()))))

    def _decode(self, session=None, nmt_model=None, data_reader=None, fin=None, fout=sys.stdout):
        # create tensorflow session, NMT model and data iterator if not specified
        tf_session = session if None != session else tf.Session(config=self._tf_config)
        if None == nmt_model: nmt_model = NmtModel(tf_session, self._config)
        if None == data_reader: data_reader = DataReader(self._config)
        # get target-side vocab ID to token dictionary
        id2vocab_dict = data_reader.get_id2vocab_dict()
        # decoding
        logging.info("%s: Begin NMT decoding at time: %s ..." \
                        % (type(self).__name__, time.asctime(time.localtime(time.time()))))
        cur_time = time.time()
        line_num = 0
        fout_att = open('att','w')
        if self._config.dump_nbest == True:
            fnbest = open('nbest','w')
        for (src, src_mask), (trg, trg_mask) in data_reader.get_next_data_batch('test'):
            # batch size must be 1
            sample_trans_ids, sample_scores, sample_atts = \
                nmt_model.beam_search_decoding(tf_session, src.T, src_mask.T, \
                self._beam_size, self._max_decoded_trg_len)
            sample_trans_ids = numpy.transpose(sample_trans_ids, [1,2,0])
            sample_atts = numpy.transpose(sample_atts, [1,2,0])
            sample_trans_ids = sample_trans_ids.tolist()
            for sample_trans_id,sample_score,sample_att in zip(sample_trans_ids,sample_scores,sample_atts):
                for i in range(len(sample_trans_id)):
                    sample_trans_id[i] = sample_trans_id[i] + [self._config.eos_id]
                    eos_pos = sample_trans_id[i].index(self._config.eos_id)
                    sample_trans_id[i] = sample_trans_id[i][:eos_pos]
                #if True == self._normalize_score_flag:
                #    # normalize score
                #    trg_lengths = numpy.array([len(t) for t in sample_trans_id])
                #    #len_penalty = numpy.power(((5. + trg_lengths) / 6.), self._config.alpha)
                #    #sample_score = numpy.array(sample_score) / len_penalty
                #    sample_score = numpy.array(sample_score) / trg_lengths
                # get best translation
                best_trans_idx = numpy.argmax(sample_score)
                sample_trans_str = self.__id2str(sample_trans_id[best_trans_idx], \
                    id2vocab_dict)
                #print >> fout, "%s" % (sample_trans_str)
                print >> fout, "%s" % (sample_trans_str.replace('@@ ', '').replace('@@', ''))
                for src_idx in sample_att[best_trans_idx]:
                    print >>fout_att,src_idx,
                print >>fout_att
                if self._config.dump_nbest == True:
                    for idx in range(len(sample_trans_id)):
                        print >>fnbest,self.__id2str(sample_trans_id[idx], id2vocab_dict)
                fout.flush()
                fout_att.flush()
            if 0 == (line_num + 1) % self._steps_per_ckpt:
                elapsed_time = time.time() - cur_time
                decode_speed = elapsed_time / float(line_num + 1)
                logging.info("%s: %d sentence decoded, elapsed time: %.4f sec, speed: %.4f sec/sent." % (type(self).__name__, line_num + 1, elapsed_time, decode_speed))
            line_num += 1
        logging.info("%s: NMT decoding completed at time: %s." \
                        % (type(self).__name__, time.asctime(time.localtime(time.time()))))
        # close tensorflow session if it is created
        if None == session:
            tf_session.close()

# private:

    def __sampling(self, session, model, sampling_num, src_dict, trg_dict, \
                                        src, src_mask, beam_size, trg_max_len, trg, trg_mask):
        src_length = src_mask.sum(1) - 1
        trg_length = trg_mask.sum(1) - 1
        for i in range(sampling_num):
            tids, score, att = model.beam_search_decoding(\
                session, src[i][None, :].T, src_mask[i][None, :].T, 1, trg_max_len)
            tids = numpy.transpose(tids, [1,2,0])
            src_str = self.__id2str(src[i][:src_length[i]], src_dict)
            ref_str = self.__id2str(trg[i][:trg_length[i]], trg_dict)
            trans_str = self.__id2str(tids[0][0][:-1], trg_dict)
            logging.info("%s: Sampling: %d" % (type(self).__name__, i))
            logging.info("%s: Source: %s" % (type(self).__name__, src_str))
            logging.info("%s: Target: %s" % (type(self).__name__, ref_str))
            logging.info("%s: Decoded: %s" % (type(self).__name__, trans_str))

    def __id2str(self, id_list, id2str_dict):
        return " ".join([id2str_dict.get(t, self._config.unk_token) for t in id_list]).strip()

    def _export_single(self, fout=sys.stdout):
      sess = tf.Session(config=self._tf_config)
      nmt_model = NmtModel(sess, self._config)
      if tf.gfile.Exists(self._config.model_dir):
        print("Reading model parameters from %s" % self._config.model_dir)
        nmt_model.exporter_saver.restore(sess, self._config.model_dir+self._config.model_path)
        #nmt_model.exporter_saver.restore(sess, "./model/train_model.ckpt9")
      else:
        print("%s not found " % self._config.model_dir)
        sys.exit();
      export = exporter.Exporter(nmt_model.exporter_saver)
      export.init(sess.graph.as_graph_def(),
        named_graph_signatures=nmt_model.signature)
      export.export(self._config.export_dir,
        tf.constant(self._config.version), sess)


def main(_):
    NmtSystemRunner().run()

if __name__ == '__main__':
    tf.app.run()

