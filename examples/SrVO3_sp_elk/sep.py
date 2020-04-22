if self.separate == False:
    # spin density or magnetization
    self.spd = self.spd[:, :, value]
    self.spd = self.spd.sum(axis=2)
    if value == [0]:
        print("Plotting spin density...")
    elif value == [1]:
        print("Plotting spin magnetization...")

    self.log.info("new spd shape =" + str(self.spd.shape))
    self.log.debug("selectIspin: ...Done")

else:
    # spin up (spin = 0) and spin down (spin = 1) separately.
    if value == [0]:
        # select spin up block
        self.spd = self.spd[:, : self.numofbands, 0]
        print("Plotting spin up bands...")

    elif value == [1]:
        # select spin down block
        self.spd = self.spd[:, self.numofbands :, 0]
        print("Plotting spin down bands...")
