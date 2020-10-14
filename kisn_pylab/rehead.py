# -*- coding: utf-8 -*-

"""

@authors: Jingyi Guo Fulgostad & bartulem (original code: B. A. Dunn)

Re-floor and re-head various tracking files according to one template file.

Due to some discrepancies in floor tilt, this script fits the floor to lie
on the plain covered by the rear point, and corrects for the elevation by
subtracting the actual offset of the sync LEDs to the floor.

This script also brings the heading parameters from several sessions to
a common footing by using a template heading session (ideally, a session
where the animal moved most naturally and exhibited the best range of
behaviors) to correct heading parameters in other comparable sessions.

!NB: the underlying assumption is that all sessions to be re-headed
with the template file had the same rigid body configuration as the
template file!

"""

import os
import sys
import time
import warnings
import numpy as np
import scipy.io
from random import shuffle
# import random
from itertools import permutations
from tqdm.notebook import tqdm
from scipy.optimize import minimize

warnings.simplefilter('ignore')


class ReHead:

    # initializer / instance attributes
    def __init__(self, template_file, other_files):
        self.template_file = template_file
        self.other_files = other_files

    # Put NANs where they belong and choose only points of interest (head points)
    def get_points(self, file_name):

        mat_file = scipy.io.loadmat(file_name)

        pdd = np.ravel(np.array(mat_file['pointdatadimensions']))
        pdd = pdd.astype(int)
        frame_number = pdd[2]

        head_origin = np.ravel(np.array(mat_file['headorigin']))
        head_origin[np.logical_or(head_origin < -100, head_origin > 100)] = np.nan
        head_origin = np.reshape(head_origin, (frame_number, 3))

        headX = np.ravel(np.array(mat_file['headX']))
        headX[np.logical_or(headX < -100, headX > 100)] = np.nan
        headX = np.reshape(headX, (frame_number, 3))

        headZ = np.ravel(np.array(mat_file['headZ']))
        headZ[np.logical_or(headZ < -100, headZ > 100)] = np.nan
        headZ = np.reshape(headZ, (frame_number, 3))

        point_data = np.ravel(np.array(mat_file['pointdata']))
        point_data[np.logical_or(point_data < -100, point_data > 100)] = np.nan
        point_data = np.reshape(point_data, (pdd[0], pdd[1], frame_number))

        # 0: ['cyan', 'first head'],
        # 1: ['dodger blue', 'second head'],
        # 2: ['lawn green', 'third head'],
        # 3: ['dark magenta', 'fourth head'],
        # 4: ['red', 'neck'],
        # 5: ['yellow', 'middle'],
        # 6: ['green', 'ass'],
        # 7: ['golden rod', 'LED Marker 1']
        # 8: ['golden rod', 'LED Marker 2']
        # 9: ['golden rod', 'LED Marker 3']

        point_number = 10  # NOTE THERE ARE MORE POINTS BUT WE DO NOT CARE ABOUT THEM
        sorted_point_data = np.empty((frame_number, point_number, 3))
        sorted_point_data[:] = np.nan
        for t in np.arange(frame_number):
            for j in np.arange(pdd[0]):
                for k in np.arange(point_number):
                    if point_data[j, :, t][3] == k:
                        sorted_point_data[t, k, :] = point_data[j, :, t][0:3]

        return mat_file, head_origin, headX, headZ, sorted_point_data

    def floor_correction(self, sorted_point_data, head_x, head_z, head_origin):
        butt_point = sorted_point_data[:, 6, :]
        butt_point = butt_point[~np.isnan(butt_point[:, 0]), :]
        nframe = len(butt_point)
        a = np.zeros(nframe)
        a[:] = 1
        b = butt_point[:, 0]
        c = butt_point[:, 1]
        X_mat = np.column_stack([a, b, c])
        y_vec = butt_point[:, 2]

        LHS = X_mat.transpose().dot(X_mat)
        RHS = X_mat.transpose().dot(y_vec)
        beta = np.linalg.solve(LHS, RHS)

        a_hat = beta[1]
        b_hat = beta[2]
        d_hat = beta[0]

        vx = np.array([1, 0, a_hat])
        vy = np.array([0, 1, b_hat])
        vz = np.array([-a_hat, -b_hat, 1])

        vx = vx / np.linalg.norm(vx)
        vy = vy / np.linalg.norm(vy)
        vz = vz / np.linalg.norm(vz)

        floor_rot_mat = np.column_stack([vx, vy, vz]).transpose()
        shpae_points = np.shape(sorted_point_data)
        sorted_point_data_new = np.zeros(shpae_points)
        sorted_point_data_new[:] = np.nan
        for t in np.arange(shpae_points[0]):
            for k in np.arange(shpae_points[1]):
                if (np.isnan(sorted_point_data[t, k, 0])):
                    continue
                sorted_point_data_new[t, k, :] = floor_rot_mat.dot(sorted_point_data[t, k, :])

        da = np.nanmedian(sorted_point_data_new[:, 7, 2])
        db = np.nanmedian(sorted_point_data_new[:, 8, 2])
        dc = np.nanmedian(sorted_point_data_new[:, 9, 2])
        led_height = (da + db + dc) / 3
        led_offset = led_height - 0.5135

        sorted_point_data_new[:, :, 2] = sorted_point_data_new[:, :, 2] - led_offset

        new_rot_mat = np.zeros((shpae_points[0], 3, 3))
        new_rot_mat[:] = np.nan
        new_head_x = np.zeros((shpae_points[0], 3))
        new_head_x[:] = np.nan
        new_head_z = np.zeros((shpae_points[0], 3))
        new_head_z[:] = np.nan
        new_head_origin = np.zeros((shpae_points[0], 3))
        new_head_origin[:] = np.nan
        for t in range(shpae_points[0]):
            if (~np.isnan(head_x[t, 0])):
                hx = head_x[t] / np.linalg.norm(head_x[t])
                hz = head_z[t] / np.linalg.norm(head_z[t])
                hy = np.cross(hz, hx)
                head_mat = np.array([hx, hy, hz])  # global to head
                new_rot_mat[t] = np.dot(floor_rot_mat, head_mat.transpose()).transpose()
                new_head_x[t] = new_rot_mat[t, 0, :]
                new_head_z[t] = new_rot_mat[t, 2, :]
                new_head_origin[t] = np.dot(floor_rot_mat, head_origin[t])

        return floor_rot_mat, sorted_point_data_new, new_head_x, new_head_z, new_head_origin

    # get a random subset of head data and check how infested it is with NANs
    def get_random_timepoints_with_four_head_points(self, head_points, check_point_num):

        """
        Inputs
        ----------
        check_point_num : int
            The number of points to estimate things with; should be minimally 300.
        ----------
        """
        # n_frames = len(head_points)
        # reshape_hps = np.reshape(head_points, (n_frames, 12))
        # total_non_nan_indices = np.where(np.sum(np.isnan(reshape_hps), axis=1) == 0)[0].tolist()
        # non_nan_indices = np.unique(random.choices(total_non_nan_indices, k=10*check_point_num))
        # if len(non_nan_indices) < check_point_num:
        #     print('There are more NANs in the test data than not!')
        #     sys.exit()
        # non_nan_indices = non_nan_indices[0:300]
        total_frame_num = len(head_points[:, 0, 0])
        time_points = list(range(total_frame_num))
        shuffle(time_points)

        non_nan_indices = []
        for i in range(10 * check_point_num):
            if np.sum(np.ravel(np.isnan(head_points[time_points[i], :, :]))) < 1:
                non_nan_indices.append(time_points[i])
                if len(non_nan_indices) >= check_point_num:
                    break

        if len(non_nan_indices) < check_point_num:
            print('There are more NANs in the test data than not!')
            sys.exit()

        return non_nan_indices

    # get edges between points
    def get_edges_between_points(self, hpts):
        ii = 0
        edges = np.zeros(6)
        for j in range(4):
            for k in range(j + 1, 4):
                ss = np.sqrt(np.sum((hpts[:, j, :] - hpts[:, k, :]) ** 2, 1))
                edges[ii] = np.median(ss)
                ii += 1

        return edges

    # find out which ordering is correct
    def which_ordering_is_correct(self, ref_edges, zero_hpts):

        other_edges = self.get_edges_between_points(hpts=zero_hpts)

        indstoedges = np.zeros((4, 4))
        ii = 0
        for j in range(4):
            for k in range(j + 1, 4):
                indstoedges[j, k] = other_edges[ii]
                indstoedges[k, j] = other_edges[ii]
                ii += 1

        def get_difference(new_order):
            try_edges = np.zeros(6)
            iii = 0
            for jj in range(4):
                for kk in range(jj + 1, 4):
                    try_edges[iii] = indstoedges[new_order[jj], new_order[kk]]
                    iii += 1
            return np.sqrt(np.sum((try_edges - ref_edges) ** 2))

        # scores = []
        # orderings = []
        # for i in range(4):
        #     for j in range(4):
        #         if j == i:
        #             continue
        #         for k in range(4):
        #             if j == k or i == k:
        #                 continue
        #             for m in range(4):
        #                 if k == m or j == m or i == m:
        #                     continue
        #                 an_order = [i, j, k, m]
        #                 print(an_order)
        #                 scores.append(get_difference(an_order))
        #                 orderings.append(an_order)
        # scores = np.ravel(np.array(scores))
        #
        # inds = np.argsort(scores)
        scores = np.zeros(24)
        orderings = list(permutations(np.arange(4)))
        for i in range(24):
            scores[i] = get_difference(orderings[i])

        inds = np.argsort(scores)

        # print('All scores', scores[inds])
        # print('Going with', min(scores), '!!!', 'which is', orderings[inds[0]], 'at', scores[inds[0]])

        return orderings[inds[0]]

    # create the transformation matrix
    def create_transformation_matrix(self, dx, dy, dz, x, y, z):

        rz = np.array([[np.cos(z), -np.sin(z), 0, 0], [np.sin(z), np.cos(z), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        ry = np.array([[np.cos(y), 0, np.sin(y), 0], [0, 1, 0, 0], [-np.sin(y), 0, np.cos(y), 0], [0, 0, 0, 1]])
        rx = np.array([[1, 0, 0, 0], [0, np.cos(x), -np.sin(x), 0], [0, np.sin(x), np.cos(x), 0], [0, 0, 0, 1]])
        r = np.dot(rx, np.dot(ry, rz))
        t = np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])

        return np.dot(t, r)

    # find mapping ans shift CSYS coordinate systems
    def find_mapping_from_A_to_B_and_shit_CSYS(self, big_a, big_b, acsys):

        def get_new_guy(big_a, vv):
            [xx, yy, zz, rotx, roty, rotz] = vv
            transformation_matrix = self.create_transformation_matrix(dx=xx, dy=yy, dz=zz, x=rotx, y=roty, z=rotz)
            new_a = np.zeros(np.shape(big_a))
            tt = len(big_a[:, 0, 0])
            nn = len(big_a[0, :, 0])
            for num in range(tt):
                for num_two in range(nn):
                    aa = np.ones(4)
                    aa[:3] = big_a[num, num_two, :]
                    new_a[num, num_two, :] = (np.dot(transformation_matrix, aa))[:3]
            return new_a

        def get_dist(vv):
            new_a = get_new_guy(big_a, vv)
            return np.sum(np.ravel((new_a - big_b) ** 2))

        opt_vv = []
        opt_score = np.nan
        for i in range(200):
            x0 = [0., 0., 0., np.random.rand() * 2. * np.pi, np.random.rand() * 2. * np.pi, np.random.rand() * 2. * np.pi]
            res = minimize(get_dist, x0, method='BFGS', tol=1e-9, options={'gtol': 1e-9, 'disp': False})
            if res.success:
                opt_vv = res.x
                opt_score = get_dist(opt_vv)
                break

        if len(opt_vv) < 1:
            return np.nan, np.nan

        res = get_new_guy(acsys, opt_vv)
        if len(res[:, 0, 0]) == 1:
            return res[0, :, :], opt_score

        return res, opt_score

    # get shifts between csys guys
    def get_shift_between_csys_guys(self, acsys, bcsys, big_a_csys):

        def csys_transformator(vv, aguy):
            orig = aguy[:, 0, :]
            xdir = aguy[:, 1, :] - aguy[:, 0, :]
            ydir = aguy[:, 2, :] - aguy[:, 0, :]
            zdir = aguy[:, 3, :] - aguy[:, 0, :]
            tcsys = np.zeros(np.shape(aguy))
            tcsys[:, 0, :] = orig + vv[0] * xdir + vv[1] * ydir + vv[2] * zdir
            tcsys[:, 1, :] = orig + vv[3] * xdir + vv[4] * ydir + vv[5] * zdir
            tcsys[:, 2, :] = orig + vv[6] * xdir + vv[7] * ydir + vv[8] * zdir
            tcsys[:, 3, :] = orig + vv[9] * xdir + vv[10] * ydir + vv[11] * zdir
            return tcsys

        def get_score(vv):
            tcsys = csys_transformator(vv, acsys)
            return np.nansum(np.ravel((tcsys - bcsys) ** 2))

        bestvv = []
        bestscore = 0.0001
        numtimesworked = 0
        for i in range(300):
            x0 = np.zeros(12)
            for j in range(12):
                x0[j] = np.random.rand() - 0.5
            res = minimize(get_score, x0, method='BFGS', tol=1e-9, options={'gtol': 1e-9, 'disp': False})
            if res.success:
                numtimesworked = numtimesworked + 1
                optvv = res.x
                score = get_score(optvv) / float(len(acsys[:, 0, 0]))
                if score < bestscore:
                    bestscore = score
                    bestvv = optvv

                if numtimesworked > 5:
                    break

        if len(bestvv) < 3:
            print('Nothing good happened here!!!')
            sys.exit()

        # print('\nBest scores was', bestscore)
        new_big_a_csys = csys_transformator(bestvv, big_a_csys)
        # tcsys = csys_transformator(bestvv, acsys)
        # xdir = tcsys[:, 1, :] - tcsys[:, 0, :]
        # ydir = tcsys[:, 2, :] - tcsys[:, 0, :]
        # zdir = tcsys[:, 3, :] - tcsys[:, 0, :]
        # shouldbeones = np.sqrt(np.sum(xdir**2, 1))
        # print('should be close to 1, x', np.nanmean(shouldbeones), np.nanstd(shouldbeones), np.nanmax(shouldbeones), np.nanmin(shouldbeones))
        # shouldbeones = np.sqrt(np.sum(ydir**2, 1))
        # print('should be close to 1, y', np.nanmean(shouldbeones), np.nanstd(shouldbeones), np.nanmax(shouldbeones), np.nanmin(shouldbeones))
        # shouldbeones = np.sqrt(np.sum(zdir**2, 1))
        # print('should be close to 1, z', np.nanmean(shouldbeones), np.nanstd(shouldbeones), np.nanmax(shouldbeones), np.nanmin(shouldbeones))

        return new_big_a_csys

    # get csys points
    def get_csys_points(self, hO, hX, hZ):
        big_t = len(hO[:, 0])
        csys = np.zeros((big_t, 4, 3))
        csys[:] = np.nan
        csys[:, 0, :] = hO
        csys[:, 1, :] = hO + hX
        csys[:, 3, :] = hO + hZ

        hY = np.zeros(np.shape(hZ))
        hY[:] = np.nan
        for t in range(big_t):
            if ~np.isnan(hX[t, 0]):
                hY[t, :] = np.cross(hZ[t, :], hX[t, :])

                if abs(sum((hY[t, :]) ** 2) - 1.) > 0.001:
                    print('WTF??? Why is Z cross X not of length 1??')
                    print('Z:', hZ[t, :])
                    print('Y:', hY[t, :])
                    print('X:', hX[t, :])
                    sys.exit()
        csys[:, 2, :] = hO + hY

        return csys

    # function that runs everything
    def conduct_transformations(self):

        # check that the files are there
        if not os.path.exists(self.template_file):
            print('Could not find file {}, try again.'.format(self.template_file))
            sys.exit()
        else:
            print('The template file is: {}'.format(self.template_file))

        print('The files to be re-headed are:')
        for file_indx, afile in enumerate(self.other_files):
            if not os.path.exists(afile):
                print('Could not find file {}, try again.'.format(afile))
                sys.exit()
            else:
                print('File number {}: {}'.format(file_indx + 1, afile))

        start_time = time.time()
        print('Re-heading file(s), please be patient - this could take >10 minutes.')

        # change name of the original file, so it's clear it's not re-headed
        rmat, headO, headX, headZ, sorted_point_data = self.get_points(file_name=self.template_file)
        headpoints = sorted_point_data[:, 0:4, :]

        os.rename(self.template_file, '{}_notreheaded.mat'.format(self.template_file[:-4]))

        reftpnts = self.get_random_timepoints_with_four_head_points(head_points=headpoints, check_point_num=300)
        refhpts = headpoints[reftpnts, :, :]
        hhO = headO[reftpnts, :]
        hhX = headX[reftpnts, :]
        hhZ = headZ[reftpnts, :]
        refcsys = self.get_csys_points(hO=hhO, hX=hhX, hZ=hhZ)

        newpts = np.zeros(np.shape(refhpts))
        for i in range(len(refhpts[:, 0, 0])):
            newpts[i, :, :] = headpoints[reftpnts[i], :, :]

        refedges = self.get_edges_between_points(hpts=refhpts)

        ii = 1
        for other_file in tqdm(self.other_files):
            Omat, OheadO, OheadX, OheadZ, Osorted_point_data = self.get_points(file_name=other_file)
            floor_rot_mat, sorted_point_data_new, new_head_x, new_head_z, new_head_origin = self.floor_correction(Osorted_point_data, OheadX, OheadZ, OheadO)
            OheadO = new_head_origin
            OheadX = new_head_x
            OheadZ = new_head_z
            Oheadpoints = sorted_point_data_new[:, 0:4, :]
            Otpnts = self.get_random_timepoints_with_four_head_points(head_points=Oheadpoints, check_point_num=300)
            Ohpts = Oheadpoints[Otpnts, :, :]
            bigOcsys = self.get_csys_points(hO=OheadO, hX=OheadX, hZ=OheadZ)
            lilOcsys = bigOcsys[Otpnts, :, :]

            ordering = self.which_ordering_is_correct(ref_edges=refedges, zero_hpts=Ohpts)
            ii += 1

            newhpts = np.zeros(np.shape(Ohpts))
            for j in range(4):
                newhpts[:, j, :] = Ohpts[:, ordering[j], :]
            Ohpts = newhpts

            newlilOcsys = np.zeros(np.shape(lilOcsys))
            newlilOcsys[:] = np.nan
            scores = np.zeros(len(Ohpts[:, 0, 0]))
            for i in range(len(Ohpts[:, 0, 0])):
                AA = np.reshape(Ohpts[i, :, :], (1, 4, 3))
                BB = np.reshape(refhpts[i, :, :], (1, 4, 3))
                Csys = np.reshape(lilOcsys[i, :, :], (1, 4, 3))
                newlilOcsys[i, :, :], scores[i] = self.find_mapping_from_A_to_B_and_shit_CSYS(big_a=AA, big_b=BB, acsys=Csys)

            # subselect the first half of the closest, good ones
            newRFcsys = refcsys[~np.isnan(scores), :, :]
            newlOcsys = newlilOcsys[~np.isnan(scores), :, :]
            newscores = scores[~np.isnan(scores)]
            inds = np.argsort(newscores)
            # print('Sorted scores, best', newscores[inds[:5]], 'and worst', newscores[inds[(-5):]])

            # only use the best half
            TT = int(round(float(len(newscores)) * 0.5))
            smallRFcsys = np.zeros((TT, 4, 3))
            smalllOcsys = np.zeros((TT, 4, 3))
            for i in range(TT):
                smallRFcsys[i, :, :] = newRFcsys[inds[i], :, :]
                smalllOcsys[i, :, :] = newlOcsys[inds[i], :, :]

            adjustedbigOcsys = self.get_shift_between_csys_guys(acsys=smalllOcsys, bcsys=smallRFcsys, big_a_csys=bigOcsys)
            adjO = adjustedbigOcsys[:, 0, :]
            xdir = adjustedbigOcsys[:, 1, :] - adjO
            ydir = adjustedbigOcsys[:, 2, :] - adjO
            zdir = adjustedbigOcsys[:, 3, :] - adjO

            NN = ~np.isnan(adjO[:, 0])
            den = np.sqrt(np.sum((xdir[NN, :]) ** 2, 1))
            for i in range(3):
                xdir[NN, i] = xdir[NN, i] / den

            den = np.sqrt(np.sum((ydir[NN, :]) ** 2, 1))
            for i in range(3):
                ydir[NN, i] = ydir[NN, i] / den

            den = np.sqrt(np.sum((zdir[NN, :]) ** 2, 1))
            for i in range(3):
                zdir[NN, i] = zdir[NN, i] / den

            newzdir = np.zeros(np.shape(zdir))
            newzdir[:] = np.nan
            newzdir[NN, :] = np.cross(xdir[NN, :], ydir[NN, :], axis=1)

            # shouldbeones = np.sum(newzdir[NN, :] * zdir[NN, :], 1)
            # print('Should be ones (dot(z,z))!!', np.mean(shouldbeones), np.std(shouldbeones), np,min(shouldbeones), np.max(shouldbeones))

            def replace_stuff(guy, name):
                if np.sum(np.ravel(np.isnan(guy))) > 0:
                    guy[np.isnan(guy)] = -100000000
                Omat[name] = np.ravel(guy)

            replace_stuff(adjO, 'headorigin')
            replace_stuff(xdir, 'headX')
            replace_stuff(ydir, 'headY')
            replace_stuff(newzdir, 'headZ')

            new_name = '{}_reheaded.mat'.format(other_file[:-4])
            scipy.io.savemat(new_name, Omat)

        print('Processing complete! It took {:.2f} minute(s).\n'.format((time.time() - start_time) / 60))
