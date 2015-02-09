guard 'shell' do
  watch(%r{^.+\.py$}) {|m| `echo #{m[0]} `}
  watch(%r{^.+\.py$}) {|m| `flake8 nlp/*.py test/*.py`}
  # watch(%r{^.+\.py$}) {|m| `tox`}
end
