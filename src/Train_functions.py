import numpy as np
import torch
import time
import random
import torch.nn.functional as F
from Masks import generate_square_subsequent_mask
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
np.set_printoptions(precision=3)


def calculate_synthetic_loss(probs=None,
                             is_next_pred=None,
                             targets=None,
                             is_next=None,
                             prob_weight=None,
                             sum_loss=True):
    """
    Calculate loss
    """
    # Find loss for missing vectors
    is_next_preds = is_next_pred.view(is_next.shape)
    is_next_sequence = F.binary_cross_entropy_with_logits(is_next_preds,
                                                          is_next,
                                                          reduction='mean')

    is_next_acc_preds = torch.round(torch.sigmoid(is_next_pred))
    is_next_acc_preds = is_next_acc_preds.view(is_next.shape)
    is_next_acc = (is_next_acc_preds == is_next).sum().float()
    is_next_acc = is_next_acc / (is_next.shape[0])

    # Find loss for altered vectors
    altered_preds = probs.view(targets.shape)
    altered_vectors = F.binary_cross_entropy_with_logits(
        altered_preds, targets.cuda())

    altered_vectors_acc = torch.round(torch.sigmoid(probs))
    altered_vectors_acc = altered_vectors_acc.view(targets.shape)
    altered_vectors_acc = (altered_vectors_acc == targets).sum().float()
    altered_vectors_acc = altered_vectors_acc / \
        (targets.shape[0]*targets.shape[1])

    if not sum_loss:
        loss = ((prob_weight * altered_vectors) +
                ((1 - prob_weight) * is_next_sequence))
    else:
        loss = is_next_sequence + altered_vectors
    return loss, is_next_sequence.item(), altered_vectors.item(
    ), is_next_acc.item() * 100, altered_vectors_acc.item() * 100


def calculate_disease_loss(preds, targets, pos_weight=None):
    preds = preds.view(targets.shape)
    if pos_weight:
        loss = F.binary_cross_entropy_with_logits(preds,
                                                  targets,
                                                  reduction='mean',
                                                  pos_weight=pos_weight)
    else:
        loss = F.binary_cross_entropy_with_logits(preds,
                                                  targets,
                                                  reduction='mean')
    preds = torch.round(torch.sigmoid(preds))
    # print(torch.mean(preds))
    # print(torch.max(preds))
    # print(torch.min(preds)
    f1 = f1_score(targets.cpu().detach(),
                  preds.cpu().detach(),
                  average='micro')
    #fpr, tpr, thresholds = roc_curve(targets.cpu().detach(), preds.cpu().detach(), pos_label=1)
    #area_under_curve = auc(fpr, tpr)
    #acc = (preds == targets).sum().float()
    #acc = acc/(targets.shape[0]*targets.shape[1])
    return loss, f1


def plot_confusion_matrix(preds, targets):
    """
    This function prints and plots the confusion matrix.
    """

    conf_mat = confusion_matrix(targets, torch.round(preds))
    f = plt.figure(dpi=300)
    ax = f.add_subplot(111)
    # annot=True to annotate cells
    sns.heatmap(conf_mat, annot=True, ax=ax, cmap="YlGnBu")

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Healthy', 'Sick'])
    ax.yaxis.set_ticklabels(['Healthy', 'Sick'])

    return ax


def plot_ROC(preds, targets):
    """
    This function prints and plots a ROC curve
    """
    fpr, tpr, thresholds = roc_curve(targets, preds)
    auc = roc_auc_score(targets, torch.round(preds))
    f = plt.figure(dpi=300)
    ax = f.add_subplot(111)
    ax.plot(fpr, tpr, label="ROC curve, auc=" + str(auc))
    ax.legend(loc=4)
    return ax


def train_model_disease(train_dataloader,
                        val_dataloader,
                        num_epochs,
                        model=None,
                        optim=None,
                        val_int=10,
                        save_as="",
                        mean=0,
                        std=1,
                        train_writer=None,
                        val_writer=None,
                        pos_weight=None):

    best_val = np.inf
    total_loss = 0
    i = 0
    n_iter = 0

    for epoch in range(num_epochs):

        epoch_loss = 0
        epoch_f1 = 0
        epoch_val_loss = 0
        epoch_val_f1 = 0
        val_iter = 0

        i += 1
        t1 = time.time()
        j = 0
        for batch in train_dataloader:
            j += 1
            # Get data
            src = batch['encoder'].cuda()
            dec = batch['decoder'].cuda()
            targets = batch['target'].cuda()
            # generate_square_subsequent_mask(dec.shape[1]).cuda()
            dec_mask = None
            src_mask = None

            preds = model(src.float(), dec.float(), src_mask, dec_mask)
            preds = preds
            optim.optimizer.zero_grad()
            loss, f1 = calculate_disease_loss(preds=preds,
                                              targets=targets,
                                              pos_weight=pos_weight)

            loss.backward()
            lr = optim.optimizer.param_groups[0]['lr']
            optim.optimizer.step()
            optim.step()

            # Write to summary
            write_summary_disease(total_loss=loss,
                                  total_f1=f1,
                                  total_lr=lr,
                                  writer=train_writer,
                                  state_str='train',
                                  i=1,
                                  n_iter=n_iter)

            epoch_loss += loss.item()
            epoch_f1 += f1
            total_loss += loss.item()

            if n_iter % val_int == 0:
                # Do a single validation run
                val_loss, val_f1, val_preds, val_trg = validate_disease(
                    model=model,
                    val_dataloader=val_dataloader,
                    train_iter=n_iter,
                    val_writer=val_writer)

                epoch_val_loss += val_loss
                epoch_val_f1 += val_f1
                val_preds = val_preds.view(val_trg.shape)

                if j == 1:
                    pr_val_trg = val_trg
                    pr_val_preds = val_preds
                else:
                    pr_val_trg = torch.cat((pr_val_trg, val_trg), 0)
                    pr_val_preds = torch.cat((pr_val_preds, val_preds), 0)
                val_iter += 1

            n_iter += 1

            t2 = time.time()
            total_time = (t2 - t1)

            # Write precission vs. recall curves to tensorboard
            write_pr_curve_disease(pr_trg=pr_val_trg,
                                   pr_pred=pr_val_preds,
                                   writer=val_writer,
                                   state_str='val',
                                   n_iter=n_iter - 1)

        epoch_val_f1 = epoch_val_f1 / val_iter
        epoch_val_loss = epoch_val_loss / val_iter

        # Print to terminal so we can see progress
        print(
            'Epoch: %i, loss = %.3f, val_loss = %.3f, train_f1_score: %.3f, val_f1_score: %.3f, Time: %.2f'
            % (i, (epoch_loss / j), epoch_val_loss, (epoch_f1 / j),
               (epoch_val_f1), total_time))

        if epoch_val_loss < best_val:
            #print('Best_val: ' +str(best_val) + ', epoch_val: ' + str(epoch_val_f1))
            best_val = epoch_val_loss
            print("Saving model.")
            torch.save(model.state_dict(), save_as + '_state_dict')
            torch.save(model, save_as)

    del src
    del targets
    del dec
    optim.optimizer.zero_grad()
    return n_iter


def train_model_synthetic(train_dataloader,
                          val_dataloader,
                          num_epochs,
                          model=None,
                          optim=None,
                          val_int=10,
                          save_as="",
                          mask_rate=0.25,
                          prob_weight=0.9,
                          mean=0,
                          std=1,
                          sum_loss=False,
                          train_writer=None,
                          val_writer=None):

    best_val = np.inf

    total_loss = 0
    i = 0
    n_iter = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_val_loss = 0
        epoch_next = 0
        epoch_next_acc = 0
        epoch_altered_acc = 0
        epoch_altered_loss = 0
        i += 1
        t1 = time.time()
        j = 0
        n_vals = 0

        val_iter = 0
        for batch in train_dataloader:
            j += 1
            # Get data
            src = batch['encoder']
            dec = batch['decoder']
            wrng_seq = batch['target']

            # generate_square_subsequent_mask(dec.shape[1]).cuda()
            dec_mask = None
            src_mask = None

            # Randomize the next sequence for is_next_sequence prediction
            dec, wrng_seq, is_next = randomize_next_sequence(dec, wrng_seq)

            # Set random x% of data to zero vector
            src, dec, targets = create_masked_LM_vectors(
                mask_rate, src, dec, wrng_seq)

            transformer_out, probs, is_next_pred = model(
                src.float(), dec.float(), src_mask, dec_mask)

            optim.optimizer.zero_grad()
            loss, next_seq, altered_loss, is_next_acc, altered_acc = calculate_synthetic_loss(
                probs=probs,
                is_next_pred=is_next_pred,
                targets=targets,
                is_next=is_next,
                prob_weight=prob_weight,
                sum_loss=sum_loss)
            loss.backward()
            optim.optimizer.step()
            optim.step()

            # Write to summary
            write_summary_synthetic(loss,
                                    next_seq,
                                    is_next_acc,
                                    altered_loss,
                                    altered_acc,
                                    writer=train_writer,
                                    state_str='train',
                                    i=1,
                                    n_iter=n_iter)

            epoch_loss += loss.item()
            epoch_next += next_seq
            epoch_next_acc += is_next_acc
            epoch_altered_acc += altered_acc
            epoch_altered_loss += altered_loss
            total_loss += loss.item()

            if n_iter % val_int == 0:
                # Do a single validation run
                val_loss, val_is_next, val_is_next_preds, val_altered_targets, val_altered_preds = validate_synthetic(
                    model=model,
                    val_dataloader=val_dataloader,
                    mask_rate=mask_rate,
                    sum_loss=sum_loss,
                    prob_weight=prob_weight,
                    mean=mean,
                    std=std,
                    train_iter=n_iter,
                    val_writer=val_writer)

                epoch_val_loss += val_loss

                if n_vals == 0:
                    pr_is_next_trg = val_is_next
                    pr_is_next_preds = val_is_next_preds
                    pr_altered_trg = val_altered_targets
                    pr_altered_preds = val_altered_preds
                else:
                    pr_is_next_trg = torch.cat((pr_is_next_trg, val_is_next),
                                               0)
                    pr_is_next_preds = torch.cat(
                        (pr_is_next_preds, val_is_next_preds), 0)
                    pr_altered_trg = torch.cat(
                        (pr_altered_trg, val_altered_targets), 0)
                    pr_altered_preds = torch.cat(
                        (pr_altered_preds, val_altered_preds), 0)

                n_vals += 1
                val_iter += 1
            n_iter += 1

            t2 = time.time()
            total_time = (t2 - t1)

        # Write precission vs. recall curves to tensorboard
        write_pr_curve_synthetic(pr_is_next_trg, pr_is_next_preds,
                                 pr_altered_trg, pr_altered_preds, val_writer,
                                 'val', n_iter - 1)

        # Print to terminal so we can see progress
        print(
            'Epoch: %i, loss = %.3f,  next_seq_loss: %.3f, next_seq_acc: %.3f, altered_loss: %.3f, altered_acc: %.3f, Time: %.2f'
            % (i, (epoch_loss / j), (epoch_next / j), (epoch_next_acc / j),
               (epoch_altered_loss / j), (epoch_altered_acc / j), total_time))

        epoch_val_loss = epoch_val_loss / val_iter
        if epoch_val_loss < best_val:
            best_val = epoch_val_loss
            print("Saving model to: " + save_as)
            torch.save(model.state_dict(), save_as + '_state_dict')
            torch.save(model, save_as)

    return n_iter


def validate_synthetic(model=None,
                       val_dataloader=None,
                       mask_rate=0.15,
                       sum_loss=False,
                       prob_weight=0.9,
                       mean=0,
                       std=1,
                       train_iter=0,
                       val_writer=None,
                       create_histogram=False):
    """
    Do a single validation run during training.
    """
    batch = iter(val_dataloader).next()
    val_s = batch['encoder']
    val_d = batch['decoder']
    val_t = batch['target']

    model.eval()
    loss, next_seq, altered_loss, next_seq_acc, altered_acc, is_next_preds, is_next_trg, altered_preds, altered_targets,\
        transformer_out = evaluate_synthetic(model,
                                             mask_rate,
                                             val_s,
                                             val_d,
                                             val_t,
                                             prob_weight,
                                             sum_loss,
                                             mean,
                                             std)

    altered_preds = altered_preds.view(altered_targets.shape)
    is_next_preds = is_next_preds.view(is_next_trg.shape)

    # Write loss and accuracy curves to tensorboard
    write_summary_synthetic(loss,
                            next_seq,
                            next_seq_acc,
                            altered_loss,
                            altered_acc,
                            writer=val_writer,
                            state_str='train',
                            i=1,
                            n_iter=train_iter)

    # Create histogram of all weights
    if create_histogram:
        for name, param in model.named_parameters():
            if 'linear' in name or 'prob' in name:
                val_writer.add_histogram(tag=name + '/train',
                                         values=param,
                                         global_step=train_iter)

    num_repeats = 50

    # Printing the actual predictions and true labels as images. I repeat vectors to create an image that is 50px high. The predictions are sigmoid activated and rounded.
    val_writer.add_images(
        'Next_seq_PRED/val',
        torch.round(torch.sigmoid(is_next_preds.view(
            is_next_preds.shape[0]))).repeat(num_repeats).view(
                num_repeats, is_next_preds.shape[0]),
        global_step=train_iter,
        dataformats='HW')

    val_writer.add_images('Next_seq_TRUE/val',
                          is_next_trg.repeat(num_repeats).view(
                              num_repeats, is_next_trg.shape[0]),
                          global_step=train_iter,
                          dataformats='HW')

    val_writer.add_images('Altered_PRED/val',
                          torch.round(
                              torch.sigmoid(
                                  altered_preds[0].repeat(num_repeats).view(
                                      num_repeats, altered_preds.shape[1]))),
                          global_step=train_iter,
                          dataformats='HW')

    val_writer.add_images('Altered_TRUE/val',
                          altered_targets[0].repeat(num_repeats).view(
                              num_repeats, altered_targets.shape[1]),
                          global_step=train_iter,
                          dataformats='HW')

    model.train()
    return loss.item(
    ), is_next_trg, is_next_preds, altered_targets, altered_preds


def validate_disease(model=None,
                     val_dataloader=None,
                     train_iter=0,
                     val_writer=None):
    """
    Do a single validation run during training.
    """
    batch = iter(val_dataloader).next()
    val_s = batch['encoder'].cuda()
    val_d = batch['decoder'].cuda()
    val_t = batch['target'].cuda()

    model.eval()
    loss, f1, preds = evaluate_disease(model, val_s, val_d, val_t)
    del val_s
    del val_d
    val_preds = preds.view(val_t.shape)

    # Write loss and accuracy curves to tensorboard
    write_summary_disease(loss,
                          f1,
                          writer=val_writer,
                          state_str='train',
                          i=1,
                          n_iter=train_iter)

    num_repeats = 50

    # Printing the actual predictions and true labels as images. I repeat vectors to create an image that is 50px high. The predictions are sigmoid activated and rounded.
    val_writer.add_images(
        'Disease_PRED/val',
        torch.round(
            torch.sigmoid(val_preds).squeeze()).repeat(num_repeats).view(
                num_repeats, val_preds.shape[0]),
        global_step=train_iter,
        dataformats='HW')

    val_writer.add_images('Disease_TARGETS/val',
                          val_t.squeeze().repeat(num_repeats).view(
                              num_repeats, val_t.shape[0]),
                          global_step=train_iter,
                          dataformats='HW')

    model.train()
    return loss.item(), f1, preds, val_t


def evaluate_synthetic(model=None,
                       mask_rate=0.25,
                       src=None,
                       dec=None,
                       wrng_seq=None,
                       prob_weight=None,
                       sum_loss=False,
                       mean=0,
                       std=1):

    # Randomize the next sequence for is_next_sequence prediction
    dec, wrng_seq, is_next = randomize_next_sequence(dec, wrng_seq)

    src, dec, altered_targets = create_masked_LM_vectors(
        mask_rate, src, dec, wrng_seq)

    # Calculate masks
    src_mask = None
    dec_mask = None  # generate_square_subsequent_mask(dec.shape[1]).cuda()

    # Pass through Transformer

    transformer_out, altered_preds, is_next_preds = model(
        src.float(), dec.float(), src_mask, dec_mask)

    # Calculate validation loss
    loss, next_seq_loss, altered_loss, next_seq_acc, altered_acc = calculate_synthetic_loss(
        probs=altered_preds,
        is_next_pred=is_next_preds,
        targets=altered_targets,
        is_next=is_next,
        prob_weight=prob_weight,
        sum_loss=sum_loss)

    return loss, next_seq_loss, altered_loss, next_seq_acc, altered_acc, is_next_preds.cpu(
    ).detach(), is_next.cpu().detach(), altered_preds.cpu().detach(
    ), altered_targets.cpu().detach(), transformer_out.cpu().detach()


def evaluate_disease(model=None, src=None, dec=None, trg=None):
    # Calculate masks
    src_mask = None
    dec_mask = None  # generate_square_subsequent_mask(dec.shape[1]).cuda()

    # Pass through Transformer
    preds = model(src.float(), dec.float(), src_mask, dec_mask)
    preds = preds

    # Calculate validation loss
    loss, f1 = calculate_disease_loss(preds=preds, targets=trg)

    return loss, f1, preds.cpu().detach()


def test_model_synthetic(test_dataloader,
                         model=None,
                         mean=0,
                         std=1,
                         mask_rate=0.15,
                         prob_weight=0.9,
                         sum_loss=False,
                         test_writer=None,
                         n_iter=0):

    # Set model to eval mode
    model.eval()

    # Prepare calculations
    total_loss = 0
    total_next_loss = 0
    total_altered_loss = 0
    total_next_seq_acc = 0
    total_altered_acc = 0
    i = 0
    print("Testing model...")
    for batch in test_dataloader:
        i += 1

        src = batch['encoder']
        dec = batch['decoder']
        wrng_seq = batch['target']

        loss, next_seq, altered_loss, next_seq_acc, altered_acc, is_next_preds, is_next, altered_preds, altered_targets,\
            transformer_out = evaluate_synthetic(model,
                                                 mask_rate,
                                                 src,
                                                 dec,
                                                 wrng_seq,
                                                 prob_weight,
                                                 sum_loss,
                                                 mean,
                                                 std)

        if i == 1:
            pr_is_next_trg = is_next
            pr_is_next_preds = is_next_preds
            pr_altered_trg = altered_targets
            pr_altered_preds = altered_preds
            emb_transformer_out = transformer_out
        else:
            pr_is_next_trg = torch.cat((pr_is_next_trg, is_next), 0)
            pr_is_next_preds = torch.cat((pr_is_next_preds, is_next_preds), 0)
            pr_altered_trg = torch.cat((pr_altered_trg, altered_targets), 0)
            pr_altered_preds = torch.cat((pr_altered_preds, altered_preds), 0)
            emb_transformer_out = torch.cat(
                (emb_transformer_out, transformer_out), 0)

        if i == 1:
            tmp_is_next = is_next
            tmp_is_next_preds = is_next_preds
        total_loss += loss.item()
        total_altered_loss += altered_loss
        total_next_loss += next_seq
        total_next_seq_acc += next_seq_acc
        total_altered_acc += altered_acc

    altered_preds = altered_preds.view(altered_targets.shape)
    tmp_is_next_preds = tmp_is_next_preds.view(tmp_is_next.shape)

    write_summary_synthetic(total_loss,
                            total_next_loss,
                            total_next_seq_acc,
                            total_altered_loss,
                            total_altered_acc,
                            pr_is_next_trg,
                            pr_is_next_preds,
                            pr_altered_trg,
                            pr_altered_preds,
                            writer=test_writer,
                            state_str='test',
                            i=i,
                            n_iter=n_iter)

    num_repeats = 50

    # Printing the actual predictions and true labels as images. I repeat vectors to create an image that is 50px high. The predictions are sigmoid activated and rounded.
    test_writer.add_images(
        'Next_seq_PRED/test',
        torch.round(
            torch.sigmoid(tmp_is_next_preds.view(
                tmp_is_next_preds.shape[0]))).repeat(num_repeats).view(
                    num_repeats, tmp_is_next_preds.shape[0]),
        global_step=n_iter,
        dataformats='HW')

    test_writer.add_images('Next_seq_TRUE/test',
                           tmp_is_next.repeat(num_repeats).view(
                               num_repeats, tmp_is_next.shape[0]),
                           global_step=n_iter,
                           dataformats='HW')

    test_writer.add_images('Altered_PRED/test',
                           torch.round(
                               torch.sigmoid(
                                   altered_preds[0].repeat(num_repeats).view(
                                       num_repeats, altered_preds.shape[1]))),
                           global_step=n_iter,
                           dataformats='HW')

    test_writer.add_images('Altered_TRUE/test',
                           altered_targets[0].repeat(num_repeats).view(
                               num_repeats, altered_targets.shape[1]),
                           global_step=n_iter,
                           dataformats='HW')

    is_next_length = min(10, is_next.shape[0])

    print('is_next_preds: ' + str(
        torch.sigmoid(is_next_preds[0:is_next_length].view(
            is_next_length)).cpu().detach().numpy()))
    print('is_next_targets: ' +
          str(is_next[0:is_next_length].cpu().detach().numpy()))

    print('')
    print(
        'Test Loss: %.3f, next_seq_loss: %.3f , next_seq_acc: %.3f, altered_loss: %.3f, altered_acc: %.3f'
        % (total_loss / i, total_next_loss / i, total_next_seq_acc / i,
           total_altered_loss / i, total_altered_acc / i))
    print('-------------------------------------------------------------')

    tmp_emb_transformer_out = emb_transformer_out[0:30]
    tmp_pr_altered_trg = pr_altered_trg[0:30]
    # Create embeddings
    embeddings = tmp_emb_transformer_out.view(
        tmp_emb_transformer_out.shape[0] * tmp_emb_transformer_out.shape[1],
        tmp_emb_transformer_out.shape[2])
    # Create label for embeddings
    labels = tmp_pr_altered_trg.view(tmp_pr_altered_trg.shape[0] *
                                     tmp_pr_altered_trg.shape[1])

    test_writer.add_embedding(tag='Transfomer_embedding/test',
                              mat=embeddings,
                              metadata=labels,
                              label_img=None,
                              global_step=n_iter)

    # Write the mcc score
    # mcc_is_next_targets = pr_is_next_trg
    # mcc_is_next_preds = torch.round(torch.sigmoid(pr_is_next_preds))
    # mcc_is_next_targets[mcc_is_next_targets == 0] = -1
    # mcc_is_next_preds[mcc_is_next_preds == 0] = -1

    # mcc_altered_targets = pr_altered_trg
    # mcc_altered_preds = torch.round(torch.sigmoid(pr_altered_preds))
    # mcc_altered_targets[mcc_altered_targets == 0] = -1
    # mcc_altered_preds[mcc_altered_preds == 0] = -1

    # test_writer.add_scalar(
    #     'MCC_is_next_score/test', matthews_corrcoef(mcc_is_next_targets.int(), mcc_is_next_preds.int()), n_iter)
    # test_writer.add_scalar(
    #     'MCC_altered_score/test', matthews_corrcoef(mcc_altered_targets.int(), mcc_altered_preds.int()), n_iter)

    return total_loss / i


def test_model_disease(test_dataloader,
                       model=None,
                       mean=0,
                       std=1,
                       test_writer=None,
                       n_iter=0):

    # Set model to eval mode
    model.eval()

    # Prepare calculations
    total_loss = 0
    total_f1 = 0
    i = 0

    pr_trg = []
    pr_preds = []

    print("Testing model...")
    for batch in test_dataloader:
        i += 1
        src = batch['encoder'].cuda()
        dec = batch['decoder'].cuda()
        targets = batch['target'].cuda()

        loss, f1, preds = evaluate_disease(model, src, dec, targets)

        if i == 1:
            pr_trg = targets.cpu().detach()
            pr_preds = preds
        else:
            pr_trg = torch.cat((pr_trg, targets.cpu().detach()), 0)
            pr_preds = torch.cat((pr_preds, preds), 0)
        total_loss += loss.item()
        total_f1 += f1
        num_repeats = 50
        # Printing the actual predictions and true labels as images. I repeat vectors to create an image that is 50px high. The predictions are sigmoid activated and rounded.
        test_writer.add_images(
            'Disease_PRED/test',
            torch.round(
                torch.sigmoid(pr_preds).squeeze()).repeat(num_repeats).view(
                    num_repeats, pr_preds.shape[0]),
            global_step=n_iter,
            dataformats='HW')

        test_writer.add_images('Disease_TARGETS/test',
                               pr_trg.squeeze().repeat(num_repeats).view(
                                   num_repeats, pr_trg.shape[0]),
                               global_step=n_iter,
                               dataformats='HW')

    # Plot ROC curve and confusion matrix
    test_writer.add_figure('Disease_Confusion_Matrix/test',
                           plot_confusion_matrix(torch.sigmoid(pr_preds),
                                                 pr_trg).get_figure(),
                           global_step=1,
                           close=True)
    test_writer.add_figure('Disease_ROC/test',
                           plot_ROC(torch.sigmoid(pr_preds),
                                    pr_trg).get_figure(),
                           global_step=1,
                           close=True)

    # Write the mcc score
    mcc_targets = pr_trg
    mcc_preds = torch.round(torch.sigmoid(pr_preds))
    mcc_targets[mcc_targets == 0] = -1
    mcc_preds[mcc_preds == 0] = -1
    test_writer.add_scalar(
        'MCC_score/test', matthews_corrcoef(mcc_targets.int(),
                                            mcc_preds.int()), n_iter)

    # Write standard summary
    write_summary_disease(total_loss=total_loss,
                          total_f1=total_f1,
                          writer=test_writer,
                          pr_trg=pr_trg,
                          pr_preds=pr_preds,
                          state_str='test',
                          i=i,
                          n_iter=n_iter)

    print('preds: ' +
          str(torch.sigmoid(preds.squeeze()).cpu().detach().numpy()))
    print('targets: ' + str((targets.squeeze()).cpu().detach().numpy()))

    print('')
    print('Test Loss: %.3f, test_f1_score: %.3f' %
          (total_loss / i, total_f1 / i))
    print('-------------------------------------------------------------')
    del src
    del dec
    del targets
    return total_loss / i


def write_summary_synthetic(total_loss,
                            next_loss,
                            next_acc,
                            altered_loss,
                            altered_acc,
                            pr_next_trg=None,
                            pr_next_pred=None,
                            pr_altered_trg=None,
                            pr_altered_pred=None,
                            writer=None,
                            state_str="",
                            i=1,
                            n_iter=1):

    writer.add_scalar('Loss_total/' + state_str, total_loss / i, n_iter)
    writer.add_scalar('Loss_next_seq/' + state_str, next_loss / i, n_iter)
    writer.add_scalar('Loss_altered/' + state_str, altered_loss / i, n_iter)
    writer.add_scalar('Accuracy_next_seq/' + state_str, next_acc / i, n_iter)
    writer.add_scalar('Accuracy_altered/' + state_str, altered_acc / i, n_iter)

    if not pr_next_trg is None:
        write_pr_curve_synthetic(pr_next_trg, pr_next_pred, pr_altered_trg,
                                 pr_altered_pred, writer, state_str, n_iter)


def write_summary_disease(total_loss,
                          total_f1,
                          total_lr=None,
                          pr_trg=None,
                          pr_preds=None,
                          writer=None,
                          state_str="",
                          i=1,
                          n_iter=1):
    writer.add_scalar('Loss_disease_total/' + state_str, total_loss / i,
                      n_iter)
    writer.add_scalar('F1_score_total/' + state_str, total_f1 / i, n_iter)

    # Plot the learning rate used
    if total_lr:
        writer.add_scalar('Learning_rate/' + state_str, total_lr, n_iter)

    if not pr_preds is None:
        write_pr_curve_disease(pr_trg, pr_preds, writer, state_str, n_iter)


def write_pr_curve_synthetic(pr_next_trg, pr_next_pred, pr_altered_trg,
                             pr_altered_pred, writer, state_str, n_iter):

    writer.add_pr_curve('PR_Curve_is_next/' + state_str, pr_next_trg,
                        torch.sigmoid(pr_next_pred.squeeze()), n_iter)
    writer.add_pr_curve('PR_Curve_altered/' + state_str, pr_altered_trg,
                        torch.sigmoid(pr_altered_pred.squeeze()), n_iter)


def write_pr_curve_disease(pr_trg,
                           pr_pred,
                           writer=None,
                           state_str="",
                           n_iter=1):

    writer.add_pr_curve('PR_curve_disease/' + state_str, pr_trg,
                        torch.sigmoid(pr_pred), n_iter)


def get_n_targets(mask_rate, seq_length):
    # The number of targets should be an int
    n_targets = int(mask_rate * seq_length)
    return n_targets


def create_masked_LM_vectors(mask_rate, src, dec, wrng_seq):
    """
    Replaces vectors in the input tensor to make a task of determining which indexes have been changed.
    """
    # Do LM masking for each of the inputs
    src, src_trg = do_LM_masking(mask_rate, src, wrng_seq, dec)
    dec, dec_trg = do_LM_masking(mask_rate, dec, wrng_seq, src)

    # Concatenate the targets to create one big target vector
    masked_targets = torch.cat((src_trg, dec_trg), dim=1)

    return src.cuda(), dec.cuda(), masked_targets.cuda()


def do_LM_masking(mask_rate, tensor_to_mask, wrng_seq, src):
    number_of_targets = get_n_targets(mask_rate, tensor_to_mask.shape[1])
    # Make a target tensor
    targets = torch.zeros(tensor_to_mask.shape[0], tensor_to_mask.shape[1])
    #zero_vector = torch.zeros(tensor_to_mask.shape[2])

    # Go over input tensor
    for batch in range(tensor_to_mask.shape[0] - 1):
        # Sample n indexes
        indexes = random.sample(list(range(0, tensor_to_mask.shape[1] - 1)),
                                number_of_targets)
        for index in indexes:
            # A random number
            rand = random.random()

            # Select a random index from the wrong tensor
            random_batch_select = random.randint(0,
                                                 tensor_to_mask.shape[0] - 1)
            random_index_in_sequence = random.randint(
                0, tensor_to_mask.shape[1] - 1)

            # Change the vector to a vector from the src_sequence 80% of the time
            if rand <= 1.0:
                # Change the vector to the random vector 20% of the time
                mask_vector = wrng_seq[random_batch_select,
                                       random_index_in_sequence]
            else:
                # Change vector to vector from src 80% of the time
                mask_vector = src[batch, random_index_in_sequence]

            # Replace the tensor with our selected replacement tensor
            tensor_to_mask[batch, index] = mask_vector

            # Set this index to one in our target
            targets[batch, index] = 1.0
    return tensor_to_mask, targets


def randomize_next_sequence(dec, wrng_seq, prob=0.5):
    """
    With a given probability, a sequence will be exhanged for a sequence not actually following the previous sequence. The is_next variable is then also set to false. This creates the is_next sequence dataset.
    """
    is_next = torch.ones(dec.shape[0])
    for i in range(dec.shape[0]):
        if random.random() < prob:
            # Find random index
            # print(wrng_seq.shape[0])
            idx = random.randint(0, wrng_seq.shape[0] - 1)
            # Store the sequence from dec
            #tmp = dec[i]
            # Replace sequence of dec with a random one
            dec[i] = wrng_seq[idx]
            # Set the
            #wrng_seq[idx] = tmp
            is_next[i] = 0.0
    return dec, wrng_seq, is_next.cuda()
