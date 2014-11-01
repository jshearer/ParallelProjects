from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker

from PLASMA.simulation import Simulation, Argument
from PLASMA.kernel import Kernel, Note, Diagnostic
from PLASMA.user import User

from PLASMA.kernels.test.TestKernels import PreSimTest,SimTest,PostSimTest
from PLASMA.kernels.Kernel_Bases import PreSimulateKernel, SimulateKernel, PostSimulateKernel
from PLASMA.arguments.RangeArgument import RangeArgument

from PLASMA import Base  # This is your declarative base class

import pytest
from pytest import raises as raises

class DatabaseTest(object):
    @classmethod
    def setup_class(cls):
        cls.engine = create_engine('sqlite:///testing.db', echo=True)
        cls.connection = cls.engine.connect()
        cls.transaction = cls.connection.begin()

        cls.session = sessionmaker()
        cls.session.configure(bind=cls.engine)
        Base.metadata.create_all(cls.connection)

    @classmethod
    def teardown_class(cls):
        cls.transaction.rollback()
        cls.connection.close()
        cls.engine.dispose()

    def setup(self):
        self.session = self.session()

    def teardown(self):
        self.session.rollback()
        self.session.close()

class TestSimulation(DatabaseTest):
    def setup(self):
        DatabaseTest.setup(self)

        self.arguments = Argument(name="test_argument",description="Argument used in testing.",data={"test":"Test argument data"})
        
        self.simulation = Simulation(self.arguments)


class TestSimulationData(DatabaseTest):

    def setup(self):
        DatabaseTest.setup(self)

        self.arguments = Argument(name="test_argument",description="Argument used in testing.",data={"test":"Test argument data"})
        
        self.simulation = Simulation(self.arguments)

    def test_kernel_stages(self):
        kernels = [PreSimulateKernel(),PreSimulateKernel(),
                   SimulateKernel(),SimulateKernel(),SimulateKernel(),
                   PostSimulateKernel()]

        for kernel in kernels:
            self.simulation.kernels.append(kernel)

        assert len(self.simulation.pre_kernels) == 2
        assert len(self.simulation.sim_kernels) == 3
        assert len(self.simulation.post_kernels) == 1

    def test_arguments_insertion(self):
        child1_data = {"child":"Hi!"}
        child_1 = Argument(description="A child argument used for testing", data=child1_data)
        child2_data = {"child":"Hi!", "test":"it works"}
        child_2 = Argument(description="A child argument used for testing", data=child2_data)

        self.arguments["child_1"] = child_1
        self.arguments["child_2"] = child_2

        assert not "test" in self.arguments["child_1"].data
        assert "test" in self.arguments["child_2"].data
        assert self.arguments["child_1"].data == child1_data
        assert self.arguments["child_2"].data == child2_data


    def test_arguments_children(self):
        child_data = {"child":"Hi!"}
        child = Argument(name="child_arg", description="A child argument used for testing", data=child_data)
        self.arguments["child_arg"] = child

        assert self.arguments["child_arg"] == child
        assert self.arguments["child_arg"].data == child_data

        #Multiple levels of recursion

        sub_child = Argument(name="child_2", description="A child argument used for testing", data=child_data)
        child["child_2"] = sub_child

        assert child["child_2"] == sub_child
        assert self.arguments["child_2"] == sub_child

    def test_defaults(self):
        assert self.simulation.step == 0
        assert self.simulation.steps == 0

    def test_range_argument(self):
        assert RangeArgument(5,range(1,10)).validate()
        assert not RangeArgument(5,range(10,100)).validate()

    def test_argument_subclass_creation(self):
        rang = [range(1,5),range(1,10),range(10,100)]

        rangearg_1 = RangeArgument(-1,rang[0])
        rangearg_2 = RangeArgument(4,rang[1])
        rangearg_3 = RangeArgument(74.5,rang[2])

        assert rangearg_1.data['range']['min'] == min(rang[0])
        assert rangearg_2.data['range']['max'] == max(rang[1])
        assert rangearg_3.data['range']['min'] == min(rang[2])
        assert rangearg_3.data['range']['max'] == max(rang[2])

    def test_recursive_validation(self):
        rangearg_1 = RangeArgument(4,range(1,10))
        rangearg_2 = RangeArgument(74.5,range(10,100))
        rangearg_3 = RangeArgument(-1,range(1,5))

        self.arguments["rangearg_1"] = rangearg_1
        self.arguments["rangearg_2"] = rangearg_2

        assert self.arguments.validate()

        self.arguments["rangearg_3"] = rangearg_3

        assert not self.arguments.validate()
        



