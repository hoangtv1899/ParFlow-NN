import json
import os.path


class PFMetadata:

    @staticmethod
    def _parse_value(v):

        # TODO: What's with the 'd0' suffix?
        if v.endswith('d0'):
            v = v[:-2]

        if v in ('True', 'False'):
            return eval(v)

        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                if ' ' in v:
                    v = [t.strip() for t in v.split(' ')]
                    v = filter(lambda item: item != '', v)
                return v
            else:
                return v
        else:
            return v

    def __init__(self, filename):
        if not os.path.exists(filename):
            raise RuntimeError(f'File {filename} not found')

        self.filename = filename
        self.config = json.loads(open(filename, 'r').read())

    def __getitem__(self, item):
        value = self.config['inputs']['configuration']['data'][item]
        return self._parse_value(value)

    def get_absolute_path(self, filename):
        # Return the absolute path of the file represented by filename
        return os.path.join(os.path.dirname(self.filename), filename)

    def nz_list(self, which):
        assert self[f'{which}.Type'] == 'nzList'
        return {i: self[f'Cell.{i}.{which}.Value'] for i in range(self[f'{which}.nzListNumber'])}

    def get_geom_values(self, which, names_field='Names'):
        d = {}
        for name in self[f'Geom.{which}.{names_field}']:
            assert self[f'Geom.{name}.{which}.Type'] == 'Constant', 'Only constants supported for now'
            d[name] = self[f'Geom.{name}.{which}.Value']
        return d

    def get_geom_by_type(self, typ):
        for geom in self['GeomInput.Names']:
            if self[f'GeomInput.{geom}.InputType'] == typ:
                return geom

    def dz_scale(self):
        if self['Solver.Nonlinear.VariableDz']:
            return self.nz_list('dzScale')
        else:
            return None

    def permeability(self):
        return self.get_geom_values('Perm', names_field='Names')

    def porosity(self):
        # TODO: Why does Porosity follow 'Geom.Porosity.GeomNames'
        # while permeability follows 'Geom.Perm.Names'
        return self.get_geom_values('Porosity', names_field='GeomNames')

    def phase_geom_values(self, phase, attribute_name):
        return {name: self[f'Geom.{name}.{phase}.{attribute_name}'] for name in self[f'Phase.{phase}.GeomNames']}

    def geom_tensors(self, which, axis):
        assert axis in ('X', 'Y', 'Z'), 'axis should be one of X/Y/Z'
        assert self[f'{which}.TensorType'] == 'TensorByGeom', 'Only TensorType = TensorByGeom supported'

        d = {}
        names = self[f'Geom.{which}.TensorByGeom.Names']
        if isinstance(names, str):  # singleton
            d[names] = self[f'Geom.{names}.{which}.TensorVal{axis}']
        else:
            for name in names:
                d[name] = self[f'Geom.{name}.{which}.TensorVal{axis}']
        return d

    def get_values_by_geom(self, which, is_reversed=False):
        assert self[f'{which}.Type'] == 'Constant'

        d = {}
        names = self[f'{which}.GeomNames']
        if isinstance(names, str):  # singleton
            if is_reversed:
                d[names] = self[f'{which}.Geom.{names}.Value']
            else:
                d[names] = self[f'Geom.{names}.{which}.Value']
        else:
            for name in names:
                if is_reversed:
                    d[name] = self[f'{which}.Geom.{name}.Value']
                else:
                    d[names] = self[f'Geom.{name}.{which}.Value']
        return d

    def indicator_file(self):
        g = self.get_geom_by_type('IndicatorField')
        return self.get_absolute_path(self[f'Geom.{g}.FileName'])

    def indicator_geom_values(self):
        g = self.get_geom_by_type('IndicatorField')
        d = {}
        for name in self[f'GeomInput.{g}.GeomNames']:
            d[name] = self[f'GeomInput.{name}.Value']
        return d

    def icpressure_refpatch(self):
        g = self['ICPressure.GeomNames']
        assert type(g) == str, 'Multiple ICPressure.GeomNames found'
        return self[f'Geom.{g}.ICPressure.RefPatch']

    def icpressure_filename(self):
        g = self['ICPressure.GeomNames']
        assert type(g) == str, 'Multiple ICPressure.GeomNames found'
        filename = self[f'Geom.{g}.ICPressure.FileName']
        assert not filename.startswith('$'), 'Not supported yet'
        return self.get_absolute_path(filename)

    def icpressure_value(self):
        g = self['ICPressure.GeomNames']
        assert type(g) == str, 'Multiple ICPressure.GeomNames found'
        return self[f'Geom.{g}.ICPressure.Value']
