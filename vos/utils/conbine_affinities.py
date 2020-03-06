
def conbine_affinity(affinities):
        # Considering the affinity could be a list of affinity, conbine all resources manually
        affinity = {f: list() for f in affinities[0].keys()}
        for k in affinity.keys():
            if "cpus" in k:
                for aff in affinities:
                    affinity[k].extend(aff[k])
            elif "torch_threads" in k:
                num = 0
                for aff in affinities:
                    num += aff[k]
                affinity[k] = num
            elif "cuda_idx" == k:
                for aff in affinities:
                    affinity[k].append(aff[k])
            else:
                # should be "alternating" and "set_affinity" keys
                affinity[k] = False
                for aff in affinities:
                    affinity[k] |= aff[k]
        return affinity