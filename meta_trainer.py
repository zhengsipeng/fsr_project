import os
import torch
from utils.utils import AverageMeter, printer, printer_cycle
from utils.visualize import vis_salient_patch


def meta_trainer(args, model, criterion, optimizer, train_loader, device, writer, tips):
    e = tips[0]
    n_iter_train = tips[1]
    id2clss = tips[2]
    losses = AverageMeter()
    train_acc = []
    model.train()
    if args.method == 'supcon':
        for idx, (datas, labels) in enumerate(train_loader):   
            # datas: 2, bsz, t, 3, h, w
            datas = torch.cat([datas, datas[:18]], dim=0)
            
            #datas = torch.cat([datas[0], datas[1]], dim=0)
            datas = datas.to(device)  # batchsize*2, T, C, H, W
            print(datas.shape)
            #assert 1==0
            labels = labels.to(device)
            bsz = labels.shape[0]

            logits, features = model(datas)

            acc = (logits.argmax(1)==labels).type(torch.cuda.FloatTensor).mean().item()
            train_acc.append(acc)
            total_acc = sum(train_acc) / len(train_acc)
            # SupCon, SimCLR
            loss, ce_loss, contrast_loss = criterion(features, logits, labels, loss_type=args.contrast_loss)
            losses.update(loss.item(), bsz)

            # update weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.use_ce:
                print('Train: [epoch: %d][iter: %d][loss: %.4f][ce: %.3f][contrast: %.3f][avg_loss: %.3f][acc: %.3f]'%(e, idx, losses.val, ce_loss.item(), contrast_loss.item(), losses.avg, total_acc))
            else:
                print('Train: [epoch: %d][iter: %d][loss: %.4f][avg_loss: %.4f]'%(e, idx, losses.val, losses.avg))
            
            torch.cuda.empty_cache()

            writer.add_scalar("Loss/train", loss.item(), n_iter_train)
            writer.add_scalar("Accuracy/train", acc, n_iter_train)
            n_iter_train += 1
    elif args.method == 'cycle':
        train_loss, train_ce, train_infonce_sp, train_infonce = [], [], [], []
        total_loss, total_ce, total_infonce_sp, total_infonce = 0, 0, 0, 0
        for i, (datas, modal_aux, labels, frames) in enumerate(train_loader):
            #params = list(model.named_parameters())
            #print(params[0][1].data)
            #print(params[-4][0])

            if i > args.epoch_iter/args.batch_size:
                break

            aux = dict()
            for modal in ['depth', 'pose', 'flow']:
                if modal in modal_aux:
                    aux[modal] = modal_aux[modal].to(device)

            datas = datas.to(device)  # batchsize, 2, T, C, H, W
            labels = labels.to(device)

            logits, st_locs, st_locs_back, p_sim_12, p_sim_21, pos_sim = model(datas, aux)
            pos_onehot = (pos_sim>args.sim_thresh).int()
            pos_num = pos_onehot.sum(1).float().mean()

            if pos_num > 120:
                args.sim_thresh += 0.01
            args.sim_thresh = max(args.sim_thresh, 0.95)

            #print(pos_onehot.sum(1))
            loss, ce_loss, infonce_sp_loss, info_nce_loss = criterion(logits, \
                        labels, st_locs, st_locs_back, p_sim_12, p_sim_21, pos_onehot)
            
            # update weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate the loss and accuracy
            train_loss.append(loss.item())
            train_ce.append(ce_loss.item())
            train_infonce_sp.append(infonce_sp_loss.item())
            train_infonce.append(info_nce_loss.item())
            total_loss = sum(train_loss)/len(train_loss)
            total_ce = sum(train_ce)/len(train_loss)
            total_infonce_sp = sum(train_infonce_sp)/len(train_loss)
            total_infonce = sum(train_infonce)/len(train_loss)

            acc = (logits.argmax(1)==labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
            train_acc.append(acc)
            total_acc = sum(train_acc) / len(train_acc)

            printer_cycle("train", e, args.num_epochs, i+1, len(train_loader), 
                           loss.item(), total_loss, ce_loss.item(), total_ce,
                           infonce_sp_loss.item(), total_infonce_sp, info_nce_loss.item(), total_infonce,
                           acc*100, total_acc*100, pos_num, args.sim_thresh)
            
            # save the checkpoint
            if i+1 % int(args.save_iter/args.batch_size) == 0:
                torch.save(model.state_dict(), os.path.join(args.save_path, "%d_%d.pth"%(e, args.epoch_iter)))
            
            # visualize
            if args.visualize:
                save_dir = args.save_path.split('/')[-1]
                vis_salient_patch(pos_sim, p_sim_21, frames, labels, id2clss, i, e, save_dir)
    else:  
        # =======================
        # Meta-learning Training
        # =======================
        print("Train... {}-way {}-shot {}-query".format(args.way, args.shot, args.query))
        train_acc = []
        train_loss = []
        for i, (datas, labels) in enumerate(train_loader):
            labels = labels.to(device)
            # prepare multi-modal auxilary
            aux = 0
            '''
            aux = dict()
            for modal in ['depth', 'pose', 'flow']:
                if modal in modal_aux:
                    aux[modal] = modal_aux[modal].to(device)
            '''
            #datas = torch.stack(datas)
         
           
            datas = datas.to(device)  # way*(shot+query), t, c, h, w
            #labels = torch.arange(args.way).repeat(args.shot+args.query).to(device)
            #query_labels = labels[args.way*args.shot:]
            logits = model(datas, aux, labels) 
            print(logits.shape, labels.shape)
            #assert 1==0
            loss = criterion(logits, labels)
         
            # calculate loss
            # onehot_labels = Variable(torch.zeros(args.way*args.query, args.way).scatter_(1, torch.arange(args.way).repeat(args.query).view(-1, 1), 1)).to(device) 
            # loss = F.mse_loss(pred, onehot_labels)

            train_loss.append(loss.item())
            total_loss = sum(train_loss)/len(train_loss)

            # update weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            # calculate accuracy
            acc = (logits.argmax(1) == labels).type(torch.cuda.FloatTensor).mean().item()
            train_acc.append(acc)
            total_acc = sum(train_acc) / len(train_acc)

            # print result
            printer("train", e, args.num_epochs, i+1, len(train_loader), loss.item(), total_loss, acc * 100, total_acc * 100)

            writer.add_scalar("Loss/train", loss.item(), n_iter_train)
            writer.add_scalar("Accuracy/train", acc, n_iter_train)
            n_iter_train += 1
 
    return n_iter_train
            
    
    
"""
if args.grad_setting == 'basic':
    loss.backward()
elif args.grad_setting == 'batch_grad':
    if  i % args.episode_per_batch == 0 and i == 0:
        batch_loss = loss
    elif i % args.episode_per_batch == 0 and i != 0:
        batch_loss.backward()
        batch_loss = loss
    else:
        batch_loss += loss
elif args.grad_setting == 'batch_mean_grad':
    loss = loss / args.episode_per_batch
"""