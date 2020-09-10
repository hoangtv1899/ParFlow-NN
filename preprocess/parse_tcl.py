class TCL:
    """
    A wrapper class to parse a tcl file for 'pfset' lines and return values as a dictionary
    """
    def __init__(self, tcl_file):
        self.tcl_file = tcl_file
        self.d = {}
        self._parse()

    def __getitem__(self, item):
        return self.d[item]

    def __iter__(self):
        return iter(self.d)

    @staticmethod
    def _parse_value(v):
        if v in ('True', 'False'):
            return eval(v)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                if ',' in v:
                    v = [t.strip() for t in v.split(',')]
                return v
            else:
                return v
        else:
            return v

    def _parse(self):
        for line in open(self.tcl_file, 'r').readlines():
            line = line.strip()
            if line.startswith('#'):
                continue

            if line.startswith('pfset'):
                tokens = line.split()
                key = tokens[1]
                if tokens[2].startswith('"'):
                    assert tokens[-1].endswith('"')
                    # Join; Remove leading and trailing quotes; Split again
                    # Return a scalar or a list depending on how many values we detect in the quotes
                    value = ' '.join(tokens[2:])[1:-1].split()
                    if len(value) > 1:
                        self.d[key] = value
                    elif len(value) == 1:
                        self.d[key] = value[0]
                    else:
                        self.d[key] = None
                elif tokens[2].startswith('['):
                    assert tokens[-1].endswith(']')
                    value = ' '.join(tokens[2:])[1:-1]
                    assert value.startswith('pfget')
                    self.d[key] = self[value.split()[1]]
                else:
                    self.d[key] = self._parse_value(tokens[2])


class Pfset:
    """
    A wrapper class for 'pfset' attributes that plays by Parflow model configuration rules
    and can return model values as scalars/list/dicts to the caller
    """
    def __init__(self, tcl_file):
        self.t = TCL(tcl_file)

    def __getitem__(self, item):
        return self.t[item]

    def __iter__(self):
        return iter(self.t)

    def nz_list(self, which):
        assert self[f'{which}.Type'] == 'nzList'
        return {i: self[f'Cell.{i}.{which}.Value'] for i in range(self[f'{which}.nzListNumber'])}

    def get_geom_dict(self, which):
        d = {}
        for name in self[f'Geom.{which}.Names']:
            assert self[f'Geom.{name}.{which}.Type'] == 'Constant', 'Only constants supported for now'
            d[name] = self[f'Geom.{name}.{which}.Value']
        return d

    def dz_scale(self):
        if self['Solver.Nonlinear.VariableDz']:
            return self.nz_list('dzScale')
        else:
            return None

    def permeability(self):
        return self.get_geom_dict('Perm')
