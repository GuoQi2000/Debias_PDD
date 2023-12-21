from src.model import DebiasModel
from src import data, exp, utils
from torch.nn.utils import clip_grad_norm_
import torch


CONFIG = 'config.yaml'

def get_recorder(args):
    if args.data in ['fever', 'fever_debiased']:
        header = ['ep', 'loss', 'dev', 'symmetric']
        types = ['d', 'e', 'f', 'f']
    elif args.data in ['mnli', 'mnli_debiased']:
        header = ['ep', 'loss', 'dev_m', 'dev_mm','dev_m_hard', 'dev_mm_hard','hans']
        types = ['d', 'e', 'f','f','f','f','f','f','f','f']
    elif args.data in ['qqp', 'qqp_debiased']:
        header = ['ep', 'loss', 'dev', 'test','paws']
        types = ['d', 'e', 'f','f','f']
    else:
        raise Exception('Unknown Dataset')
    return exp.DataLogger(folder = args.folder, header = header, types = types)

TEST = 0
CONFIG = './config.yaml'

def main():
    args = exp.load_args(CONFIG)
    logger = utils.EventLogger(args.folder, True)
    if args.debug:
        logger.info('='*30+'DEBUGGING'+'='*30)
    exp.set_seed(args.seed)

    if args.data in ['qqp','qqp_debiased']:
        args.out_dim = 2
    else:
        args.out_dim = 3
        
    # build model
    logger.info('initializing bert classifier')
    if args.shuffle_which == 'hyp':
        bias_tokenizer = get_tokenizer(model_version='mnli-roberta-base')
        bias_net = get_bias_model(model_version='mnli-roberta-base_h')
        net = DebiasModel(args,bias_model=bias_net,bias_tokenizer=bias_tokenizer).cuda(args.cuda)
    else:
        net = DebiasModel(args).cuda(args.cuda)

    # read data
    if args.data in ['mnli_debiased','mnli']:
        train, dev_m,dev_mm,dev_m_hard,dev_mm_hard,hans,test_m,test_mm,test_m_hard,test_mm_hard = data.get_train_dev_test_set(args.data)
        print('train: ',len(train[0]))
        print('dev_m: ',len(dev_m[0]))
        print('dev_mm: ',len(dev_mm[0]))
        print('dev_m_hard: ',len(dev_m_hard[0]))
        print('dev_mm_hard: ',len(dev_mm_hard[0]))
        print('hans: ',len(hans[0]))
        print('test_m: ',len(test_m[0]))
        print('test_mm: ',len(test_mm[0]))
        print('test_m_hard: ',len(test_m_hard[0]))
        print('test_mm_hard: ',len(test_mm_hard[0]))
    elif args.data in ['qqp_debiased','qqp']:
        train, dev,test,paws = data.get_train_dev_test_set(args.data)
        print('train: ',len(train[0]))
        print('dev:   ',len(dev[0]))
        print('test:  ',len(test[0]))
        print('paws:  ',len(paws[0]))
    elif args.data in ['fever_debiased','fever']:
        train, dev, test = data.get_train_dev_test_set(args.data)
        print('train: ',len(train[0]))
        print('dev:   ',len(dev[0]))
        print('symmetric:  ',len(test[0]))
    else:
        print(f'Unkown task: {args.data}')
    if args.debug:
        train = [s[:10*args.batch_size] for s in train]

    # test iterator
    if args.data in ['mnli_debiased','mnli']:
        train_iter = net.build_data_iterator(*train, shuffle = True, data=args.data)

        dev_m_iter = net.build_data_iterator(*dev_m, shuffle = False, data='mnli_dev_m')
        dev_mm_iter = net.build_data_iterator(*dev_mm, shuffle = False, data='mnli_dev_mm')
        dev_m_hard_iter = net.build_data_iterator(*dev_m_hard, shuffle = False, data='mnli_dev_m_hard')
        dev_mm_hard = net.build_data_iterator(*dev_mm_hard, shuffle = False, data='mnli_dev_mm_hard')

        hans_iter = net.build_data_iterator(*hans, shuffle = False, data='hans')

        test_m_iter = net.build_data_iterator(sent1=test_m[1], sent2 = test_m[2], labels=None, shuffle = False, data='mnli_test_m')
        test_mm_iter = net.build_data_iterator(sent1=test_mm[1], sent2 = test_mm[2], labels=None, shuffle = False, data='mnli_test_mm')
        test_m_hard_iter = net.build_data_iterator(sent1=test_m_hard[1], sent2 = test_m_hard[2], labels=None,shuffle = False, data='mnli_test_m_hard')
        test_mm_hard_iter = net.build_data_iterator(sent1=test_mm_hard[1], sent2 = test_mm_hard[2], labels=None, shuffle = False, data='mnli_test_mm_hard')

        t_iters = [dev_m_iter, dev_mm_iter,dev_m_hard_iter,dev_mm_hard,hans_iter]
        p_iters = [test_m_iter, test_mm_iter, test_m_hard_iter, test_mm_hard_iter]
    elif args.data in ['fever_debiased','fever']:
        train_iter = net.build_data_iterator(*train, shuffle = True, data=args.data)
        dev_iter = net.build_data_iterator(*dev, shuffle = False, data='fever_dev')
        test_iter = net.build_data_iterator(*test, shuffle = False, data='fever-symmetric')
        t_iters = [dev_iter, test_iter]
    elif args.data in ['qqp_debiased', 'qqp']:
        train_iter = net.build_data_iterator(*train, shuffle = True, data=args.data)
        dev_iter = net.build_data_iterator(*dev, shuffle = False, data='qqp_dev')
        test_iter = net.build_data_iterator(*test, shuffle = False, data='qqp_test')
        paws_iter = net.build_data_iterator(*paws, shuffle = False, data='paws')
        t_iters = [dev_iter, test_iter, paws_iter]
    else:
        raise Exception("UNKNOWN dataset")

    # optimizer parameters
    args.total_steps = len(train_iter) * args.epochs
    args.warmup_steps = 2000
    
    losses = []
    logger.info(args.__dict__)
    optimizer, scheduler = net.setup_optimizers()
    # setup optimizer
    recorder = get_recorder(args)
    #saver = exp.Saver(net, args, optimizer, scheduler, file_name='checkpoint.pt')
    for ep in range(args.epochs):
        print('Finetuning')
        total_loss = 0
        num = 0
        net.train()
        for loss in train_iter:
            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm > 0:
                clip_grad_norm_(net.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss))
            total_loss += float(loss)
            num += 1
        net.eval()

        # evaluation
        row = [ep, total_loss/num]
        for it in t_iters:
            logits, ys, acc = it.infer()
            if it.data in ['hans', 'fever-symmetric', 'paws']:
                d = {
                    'logits': logits.cpu(),
                    'ys': ys.cpu(),
                    'acc': acc
                }
                # torch.save(d, utils.join_path(args.folder, f'test-logits-{ep}.pt'))
            row.append(acc)
        recorder.print_and_log(row)
        
        # prediction
        # if args.data in ['mnli_debiased','mnli']:
        #     folders = ['test_m','test_mm','test_m_hard','test_mm_hard']
        #     label_map = {0:'entailment',1:'neutral',2:'contradiction'}
        #     pairIDs = [test_m[0],test_mm[0],test_m_hard[0],test_mm_hard[0]]
        #     for k in range(len(p_iters)):
        #         it = p_iters[k]
        #         logits, preds = it.infer()
        #         folder = folders[k]
        #         pairID = pairIDs[k]
        #         with open(utils.join_path(args.folder, f'{folder}-{ep}.csv'),'a+') as f:
        #             f.write('pairID,gold_label'+'\n')
        #             if folder == 'test_m' or folder == 'test_mm':
        #                 for j in range(preds.shape[0]):
        #                     f.write(pairID[j]+','+label_map[int(preds[j])]+'\n')
        #             elif folder == 'test_m_hard':
        #                 init_id = 9847
        #                 for j in range(preds.shape[0]):
        #                     while(init_id < int(pairID[j])):
        #                         f.write(str(init_id)+','+'none'+'\n')
        #                         init_id += 1
        #                     f.write(pairID[j]+','+label_map[int(preds[j])]+'\n')
        #                     init_id+=1
        #             else:
        #                 init_id = 0
        #                 for j in range(preds.shape[0]):
        #                     while(init_id < int(pairID[j])):
        #                         f.write(str(init_id)+','+'none'+'\n')
        #                         init_id += 1
        #                     f.write(pairID[j]+','+label_map[int(preds[j])]+'\n')
        #                     init_id+=1
        #             f.close()
    # save the last model
    # saver = exp.Saver(net, args, optimizer, scheduler, file_name='last.pt')
    # saver.save(-1)

    
if __name__ == '__main__':
    main()