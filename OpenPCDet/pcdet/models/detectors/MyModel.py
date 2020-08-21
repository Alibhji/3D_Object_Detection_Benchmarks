from .detector3d_template import Detector3DTemplate


class MyModel(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()


    def forward(self, batch_dict):
        print(self.module_list)
        ii=0
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
            print(ii ,batch_dict.keys(),'\n')
            ii=ii+1
        # print("result------>" , batch_dict.keys())
        print("point_features------>", batch_dict['point_features'].shape)
        print("image2_features------>", batch_dict['image2_features'].shape)
        stop

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
